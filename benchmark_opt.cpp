#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <chrono>
#include <omp.h>

// Combined box filter: uint8→float mu (E[X]) and cr (E[X²]) in one pass.
// Accumulates in int32_t (fits 205×255=52275 for sum, 205×65025≈13M for
// sum-of-squares — still within int32_t range).
// Uses 3-way loop split to eliminate min/max in the hot path.
static void box_filter_row_u8_mu_cr(const uint8_t* in, float* mu, float* cr, int W, int r) {
    const float s = 1.0f / (2 * r + 1);
    if (W <= 0) return;
    int32_t sx = 0, sx2 = 0;
    for (int i = -r; i <= r; ++i) {
        uint8_t v = in[std::max(0, std::min(W - 1, i))];
        sx += v; sx2 += v * v;
    }
    mu[0] = sx * s; cr[0] = sx2 * s;
    int x = 1;
    int le = (r + 1 < W) ? (r + 1) : W;
    for (; x < le; ++x) {
        uint8_t l = in[0], rv = in[x + r];
        sx += rv - l; sx2 += rv*rv - l*l;
        mu[x] = sx * s; cr[x] = sx2 * s;
    }
    int me = (W - r > le) ? (W - r) : le;
    for (; x < me; ++x) {
        uint8_t l = in[x - r - 1], rv = in[x + r];
        sx += rv - l; sx2 += rv*rv - l*l;
        mu[x] = sx * s; cr[x] = sx2 * s;
    }
    for (; x < W; ++x) {
        uint8_t l = in[x - r - 1], rv = in[W - 1];
        sx += rv - l; sx2 += rv*rv - l*l;
        mu[x] = sx * s; cr[x] = sx2 * s;
    }
}

// Float box filter for a/b coefficients in the full guided filter.
static void box_filter_row_f32(const float* in, float* out, int W, int r) {
    const float s = 1.0f / (2 * r + 1);
    if (W <= 0) return;
    float sum = 0;
    for (int i = -r; i <= r; ++i) sum += in[std::max(0, std::min(W - 1, i))];
    out[0] = sum * s;
    int x = 1;
    int le = (r + 1 < W) ? (r + 1) : W;
    for (; x < le; ++x) { sum += in[x + r] - in[0]; out[x] = sum * s; }
    int me = (W - r > le) ? (W - r) : le;
    for (; x < me; ++x) { sum += in[x + r] - in[x - r - 1]; out[x] = sum * s; }
    for (; x < W; ++x)  { sum += in[W - 1] - in[x - r - 1]; out[x] = sum * s; }
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------
static void interpolate_offsets(float* off, const char* valid, int W) {
    int fv = -1;
    for (int x = 0; x < W; ++x) { if (valid[x]) { fv = x; break; } }
    if (fv == -1) { std::fill(off, off + W, 0.0f); return; }
    for (int x = 0; x < fv; ++x) off[x] = off[fv];
    int lv = fv;
    for (int x = fv+1; x < W; ++x) {
        if (valid[x]) {
            float y0=off[lv], y1=off[x], id=1.0f/(x-lv);
            for (int k = lv+1; k < x; ++k) off[k] = y0 + (k-lv)*id*(y1-y0);
            lv = x;
        }
    }
    for (int x = lv+1; x < W; ++x) off[x] = off[lv];
}

// ---------------------------------------------------------------------------
// Solution 1 — Destructive simplification: guided filter without the
// second pair of box filters on a/b, no rf buffer, no clamping.
// Saves 2 box filter calls per row vs the full guided filter.
// ---------------------------------------------------------------------------
static void destripe_1(const uint8_t* img, float* dst,
                       int W, int H, int r, float eps) {
    std::vector<float> off(W, 0.0f);

    #pragma omp parallel
    {
        std::vector<float> mu(W), cr(W), a(W);
        std::vector<float> loc(W, 0.0f);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = img + y * W;
            box_filter_row_u8_mu_cr(row, mu.data(), cr.data(), W, r);
            for (int i = 0; i < W; ++i) {
                float v = cr[i] - mu[i] * mu[i];
                a[i] = v / (v + eps);
            }
            for (int i = 0; i < W; ++i) {
                float v = (float)row[i];
                loc[i] += (1.0f - a[i]) * (v - mu[i]);
            }
        }

        #pragma omp critical
        for (int x = 0; x < W; ++x) off[x] += loc[x];
    }

    float iH = 1.0f / H;
    for (int x = 0; x < W; ++x) off[x] *= iH;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const uint8_t* ir = img + y*W;
        float* dr = dst + y*W;
        for (int x = 0; x < W; ++x) dr[x] = (float)ir[x] - off[x];
    }
}

// ---------------------------------------------------------------------------
// Solution 2 — Full guided filter + simple horizontal gradient mask.
// ---------------------------------------------------------------------------
static void destripe_2(const uint8_t* img, float* dst,
                       int W, int H, int r, float eps,
                       float grad_thresh, int min_flat) {
    std::vector<float> off(W, 0.0f);
    std::vector<int> cnt(W, 0);
    std::vector<char> valid(W, 0);
    int ithresh = (int)grad_thresh;

    #pragma omp parallel
    {
        std::vector<float> mu(W), cr(W), a(W), b(W), ma(W), mb(W);
        std::vector<float> lsum(W, 0.0f);
        std::vector<int> lcnt(W, 0);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = img + y*W;
            box_filter_row_u8_mu_cr(row, mu.data(), cr.data(), W, r);
            for (int i = 0; i < W; ++i) {
                float v = cr[i] - mu[i]*mu[i];
                a[i] = v / (v + eps);
                b[i] = mu[i] - a[i] * mu[i];
            }
            box_filter_row_f32(a.data(), ma.data(), W, r);
            box_filter_row_f32(b.data(), mb.data(), W, r);

            if (y > 0 && y < H - 1) {
                int x = 0;
                if (std::abs((int)row[0] - (int)row[1]) < ithresh) {
                    float v = (float)row[0];
                    lsum[0] += v - (ma[0] * v + mb[0]);
                    ++lcnt[0];
                }
                for (x = 1; x < W - 1; ++x) {
                    if (std::abs((int)row[x] - (int)row[x-1]) < ithresh
                     && std::abs((int)row[x] - (int)row[x+1]) < ithresh) {
                        float v = (float)row[x];
                        lsum[x] += v - (ma[x] * v + mb[x]);
                        ++lcnt[x];
                    }
                }
                if (std::abs((int)row[W-1] - (int)row[W-2]) < ithresh) {
                    float v = (float)row[W-1];
                    lsum[W-1] += v - (ma[W-1] * v + mb[W-1]);
                    ++lcnt[W-1];
                }
            }
        }

        #pragma omp critical
        for (int x = 0; x < W; ++x) { off[x] += lsum[x]; cnt[x] += lcnt[x]; }
    }

    for (int x = 0; x < W; ++x) {
        if (cnt[x] >= min_flat) { off[x] /= cnt[x]; valid[x] = 1; }
        else off[x] = 0.0f;
    }

    interpolate_offsets(off.data(), valid.data(), W);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const uint8_t* ir = img + y*W;
        float* dr = dst + y*W;
        for (int x = 0; x < W; ++x) dr[x] = (float)ir[x] - off[x];
    }
}

// ---------------------------------------------------------------------------
// Solution 3 — Full weighted guided filter + simple horizontal gradient mask.
// ---------------------------------------------------------------------------
static void destripe_3(const uint8_t* img, float* dst,
                       int W, int H, int r, float eps, float eta,
                       float grad_thresh, int min_flat) {
    std::vector<float> off(W, 0.0f);
    std::vector<int> cnt(W, 0);
    std::vector<char> valid(W, 0);
    int ithresh = (int)grad_thresh;

    #pragma omp parallel
    {
        std::vector<float> mu(W), cr(W), a(W), b(W), ma(W), mb(W), var(W);
        std::vector<float> lsum(W, 0.0f);
        std::vector<int> lcnt(W, 0);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = img + y*W;
            box_filter_row_u8_mu_cr(row, mu.data(), cr.data(), W, r);

            float sv = 0;
            for (int i = 0; i < W; ++i) {
                float v = cr[i] - mu[i]*mu[i];
                var[i] = v; sv += v;
            }
            float mv = sv / W;
            for (int i = 0; i < W; ++i) {
                float chi = (var[i] + eta) / (mv + eta);
                a[i] = var[i] / (var[i] + eps / chi);
                b[i] = mu[i] - a[i] * mu[i];
            }
            box_filter_row_f32(a.data(), ma.data(), W, r);
            box_filter_row_f32(b.data(), mb.data(), W, r);

            if (y > 0 && y < H - 1) {
                int x = 0;
                if (std::abs((int)row[0] - (int)row[1]) < ithresh) {
                    float v = (float)row[0];
                    lsum[0] += v - (ma[0] * v + mb[0]);
                    ++lcnt[0];
                }
                for (x = 1; x < W - 1; ++x) {
                    if (std::abs((int)row[x] - (int)row[x-1]) < ithresh
                     && std::abs((int)row[x] - (int)row[x+1]) < ithresh) {
                        float v = (float)row[x];
                        lsum[x] += v - (ma[x] * v + mb[x]);
                        ++lcnt[x];
                    }
                }
                if (std::abs((int)row[W-1] - (int)row[W-2]) < ithresh) {
                    float v = (float)row[W-1];
                    lsum[W-1] += v - (ma[W-1] * v + mb[W-1]);
                    ++lcnt[W-1];
                }
            }
        }

        #pragma omp critical
        for (int x = 0; x < W; ++x) { off[x] += lsum[x]; cnt[x] += lcnt[x]; }
    }

    for (int x = 0; x < W; ++x) {
        if (cnt[x] >= min_flat) { off[x] /= cnt[x]; valid[x] = 1; }
        else off[x] = 0.0f;
    }

    interpolate_offsets(off.data(), valid.data(), W);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const uint8_t* ir = img + y*W;
        float* dr = dst + y*W;
        for (int x = 0; x < W; ++x) dr[x] = (float)ir[x] - off[x];
    }
}

// ===================================================================
int main() {
    const int W=2048, H=2048, sz=W*H;

    // Load PNG directly as uint8_t — no float32 conversion of the full image
    // First read via a small helper: load as float .raw then convert
    // (since we don't have a PNG decoder in C++)
    std::vector<float> img_f32(sz);
    std::ifstream("hikrobot.raw",std::ios::binary)
        .read((char*)img_f32.data(),sz*4);

    // Convert to uint8_t once — this is the only full image conversion
    std::vector<uint8_t> img(sz);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < sz; ++i) img[i] = static_cast<uint8_t>(std::round(img_f32[i]));

    std::vector<float> d1(sz), d2(sz), d3(sz);

    const int r=102; const float eps=50,eta=10,gt=3,mo=3; const int mf=10;

    #pragma omp parallel
    { volatile int _ = omp_get_thread_num(); (void)_; }

    std::cout << omp_get_max_threads() << " threads\n"
              << "uint8_t row-streaming (int32 box filter accumulation)\n";

    const int N = 10;
    std::cout << "Benchmark (avg " << N << ", " << W << "x" << H << ")\n\n";

    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;++i) destripe_1(img.data(),d1.data(),W,H,r,eps);
    auto t1=std::chrono::high_resolution_clock::now();
    double s1=std::chrono::duration<double,std::milli>(t1-t0).count()/N;

    t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;++i) destripe_2(img.data(),d2.data(),W,H,r,eps,gt,mf);
    t1=std::chrono::high_resolution_clock::now();
    double s2=std::chrono::duration<double,std::milli>(t1-t0).count()/N;

    t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;++i) destripe_3(img.data(),d3.data(),W,H,r,eps,eta,gt,mf);
    t1=std::chrono::high_resolution_clock::now();
    double s3=std::chrono::duration<double,std::milli>(t1-t0).count()/N;

    double b1=44.06,b2=63.58,b3=64.35;
    std::cout << "=========  RESULTS (uint8_t row-streaming)  =========\n"
              << "S1 (Sui Orig):       " << s1 << " ms  (" << (b1/s1) << "x)\n"
              << "S2 (Masked GF):      " << s2 << " ms  (" << (b2/s2) << "x)\n"
              << "S3 (Masked WGF):     " << s3 << " ms  (" << (b3/s3) << "x)\n";

    std::ofstream("s1_opt.raw",std::ios::binary).write((char*)d1.data(),sz*4);
    std::ofstream("s2_opt.raw",std::ios::binary).write((char*)d2.data(),sz*4);
    std::ofstream("s3_opt.raw",std::ios::binary).write((char*)d3.data(),sz*4);
    std::cout << "\nDone.\n";
    return 0;
}
