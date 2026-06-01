#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <chrono>
#include <omp.h>

// ---------------------------------------------------------------------------
// Box filter on uint8_t input → float output.
// Accumulates in int32_t (fits 205 × 255 = 52275), only converts to float
// at the final multiply — much faster than float accumulation.
// Uses 3-way loop split to eliminate min/max in the hot path.
// ---------------------------------------------------------------------------
static void box_filter_row_u8(const uint8_t* in, float* out, int W, int r) {
    const float s = 1.0f / (2 * r + 1);
    if (W <= 0) return;

    int32_t sum = 0;
    for (int i = -r; i <= r; ++i) sum += in[std::max(0, std::min(W - 1, i))];
    out[0] = sum * s;

    int x = 1;
    int le = (r + 1 < W) ? (r + 1) : W;
    for (; x < le; ++x) { sum += in[x + r] - in[0]; out[x] = sum * s; }
    int me = (W - r > le) ? (W - r) : le;
    for (; x < me; ++x) { sum += in[x + r] - in[x - r - 1]; out[x] = sum * s; }
    for (; x < W; ++x)  { sum += in[W - 1] - in[x - r - 1]; out[x] = sum * s; }
}

// float → float box filter (for a, b coefficients)
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
// 5×5 separable Gaussian — uint8_t input, int32 accumulation in horizontal
// pass, float temp for vertical pass.
// ---------------------------------------------------------------------------
static void gaussian_5x5_u8(const uint8_t* src, float* dst, int W, int H) {
    std::vector<float> tmp(W * H);
    const float inv16 = 1.0f / 16.0f;

    // Horizontal — accumulate in int32, divide to float
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const uint8_t* r = src + y * W;
        float* t = tmp.data() + y * W;
        int x = 0;
        // left border
        for (; x < 2 && x < W; ++x) {
            int xm2 = 0, xm1 = 0, xp1 = (x+1<W)?x+1:W-1, xp2 = (x+2<W)?x+2:W-1;
            t[x] = (r[xm2] + 4*r[xm1] + 6*r[x] + 4*r[xp1] + r[xp2]) * inv16;
        }
        // main
        for (; x < W-2; ++x)
            t[x] = (r[x-2] + 4*r[x-1] + 6*r[x] + 4*r[x+1] + r[x+2]) * inv16;
        // right border
        for (; x < W; ++x) {
            int xm2 = (x-2>=0)?x-2:0, xm1 = (x-1>=0)?x-1:0;
            int xp1 = (x+1<W)?x+1:W-1, xp2 = (x+2<W)?x+2:W-1;
            t[x] = (r[xm2] + 4*r[xm1] + 6*r[x] + 4*r[xp1] + r[xp2]) * inv16;
        }
    }

    // Vertical — float on float
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* t = tmp.data();
        float* d = dst + y * W;
        if (y < 2 || y >= H-2) {
            int ym2 = std::max(0, y-2), ym1 = std::max(0, y-1);
            int yp1 = std::min(H-1, y+1), yp2 = std::min(H-1, y+2);
            for (int i = 0; i < W; ++i)
                d[i] = (t[ym2*W+i] + 4*t[ym1*W+i] + 6*t[y*W+i]
                      + 4*t[yp1*W+i] + t[yp2*W+i]) * inv16;
        } else {
            const float* t0 = t + (y-2)*W, *t1 = t + (y-1)*W;
            const float* t2 = t + y*W,     *t3 = t + (y+1)*W;
            const float* t4 = t + (y+2)*W;
            for (int i = 0; i < W; ++i)
                d[i] = (t0[i] + 4*t1[i] + 6*t2[i] + 4*t3[i] + t4[i]) * inv16;
        }
    }
}

// ---------------------------------------------------------------------------
// Flat mask — Sobel on Gaussian-blurred uint8_t
// ---------------------------------------------------------------------------
static void compute_flat_mask_u8(const uint8_t* img, char* mask,
                                 int W, int H, float grad_thresh) {
    std::vector<float> bl(W * H);
    gaussian_5x5_u8(img, bl.data(), W, H);

    if (H > 0) { std::fill(mask, mask + W, 0); std::fill(mask + (H-1)*W, mask + H*W, 0); }

    float t2 = grad_thresh * grad_thresh * 16.0f;
    #pragma omp parallel for schedule(static)
    for (int y = 1; y < H-1; ++y) {
        const float* r0 = bl.data() + (y-1)*W;
        const float* r1 = bl.data() + y*W;
        const float* r2 = bl.data() + (y+1)*W;
        char* m = mask + y*W;
        // left border
        for (int x = 0; x < 2 && x < W; ++x) {
            int xm1 = (x>0?x-1:0), xp1 = (x+1<W?x+1:W-1);
            float gx = (r0[xp1]+2*r1[xp1]+r2[xp1]) - (r0[xm1]+2*r1[xm1]+r2[xm1]);
            float gy = (r2[xm1]+2*r2[x]+r2[xp1]) - (r0[xm1]+2*r0[x]+r0[xp1]);
            m[x] = (gx*gx+gy*gy < t2) ? 1 : 0;
        }
        // main
        for (int x = 2; x < W-2; ++x) {
            float gx = (r0[x+1]+2*r1[x+1]+r2[x+1]) - (r0[x-1]+2*r1[x-1]+r2[x-1]);
            float gy = (r2[x-1]+2*r2[x]+r2[x+1]) - (r0[x-1]+2*r0[x]+r0[x+1]);
            m[x] = (gx*gx+gy*gy < t2) ? 1 : 0;
        }
        // right border
        for (int x = (W-2>2?W-2:2); x < W; ++x) {
            int xm1 = x-1, xp1 = (x+1<W?x+1:W-1);
            float gx = (r0[xp1]+2*r1[xp1]+r2[xp1]) - (r0[xm1]+2*r1[xm1]+r2[xm1]);
            float gy = (r2[xm1]+2*r2[x]+r2[xp1]) - (r0[xm1]+2*r0[x]+r0[xp1]);
            m[x] = (gx*gx+gy*gy < t2) ? 1 : 0;
        }
    }
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
// Solution 1 — uint8_t row-streaming, no full smooth buffer.
// Guided filter + column accumulation are combined into a single row
// pass, eliminating the 16 MB smooth buffer write+read.
// ---------------------------------------------------------------------------
static void destripe_1(const uint8_t* img, float* dst,
                       int W, int H, int r, float eps, float max_off) {
    std::vector<float> off(W, 0.0f);

    // Combined guided filter + column offset accumulation
    // (one row pass instead of two)
    #pragma omp parallel
    {
        std::vector<float> rf(W), r2(W), mu(W), cr(W), a(W), b(W), ma(W), mb(W);
        std::vector<float> loc(W, 0.0f);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = img + y * W;

            for (int i = 0; i < W; ++i) { float v = row[i]; rf[i] = v; r2[i] = v * v; }
            box_filter_row_u8(row, mu.data(), W, r);
            box_filter_row_f32(r2.data(), cr.data(), W, r);
            for (int i = 0; i < W; ++i) {
                float v = cr[i] - mu[i] * mu[i];
                a[i] = v / (v + eps);
                b[i] = (1.0f - a[i]) * mu[i];
            }
            box_filter_row_f32(a.data(), ma.data(), W, r);
            box_filter_row_f32(b.data(), mb.data(), W, r);

            // smooth output + accumulate column diffs in one row pass
            for (int i = 0; i < W; ++i) {
                float s = ma[i] * rf[i] + mb[i];
                loc[i] += rf[i] - s;
            }
        }

        #pragma omp critical
        for (int x = 0; x < W; ++x) off[x] += loc[x];
    }

    float iH = 1.0f/H;
    for (int x = 0; x < W; ++x)
        off[x] = std::max(-max_off, std::min(max_off, off[x]*iH));

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const uint8_t* ir = img + y*W;
        float* dr = dst + y*W;
        for (int x = 0; x < W; ++x) dr[x] = (float)ir[x] - off[x];
    }
}

// ---------------------------------------------------------------------------
// Solution 2 — uint8_t row-streaming, combined guided filter + accumulation
// ---------------------------------------------------------------------------
static void destripe_2(const uint8_t* img, float* dst,
                       int W, int H, int r, float eps,
                       float grad_thresh, int min_flat, float max_off) {
    std::vector<char> mask(W*H);
    compute_flat_mask_u8(img, mask.data(), W, H, grad_thresh);

    std::vector<float> off(W, 0.0f);
    std::vector<int> cnt(W, 0);
    std::vector<char> valid(W, 0);

    #pragma omp parallel
    {
        std::vector<float> rf(W), r2(W), mu(W), cr(W), a(W), b(W), ma(W), mb(W);
        std::vector<float> lsum(W, 0.0f);
        std::vector<int> lcnt(W, 0);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = img + y*W;
            const char* mr = mask.data() + y*W;

            for (int i = 0; i < W; ++i) { float v = row[i]; rf[i] = v; r2[i] = v*v; }
            box_filter_row_u8(row, mu.data(), W, r);
            box_filter_row_f32(r2.data(), cr.data(), W, r);
            for (int i = 0; i < W; ++i) {
                float v = cr[i] - mu[i]*mu[i];
                a[i] = v / (v + eps);
                b[i] = (1.0f - a[i]) * mu[i];
            }
            box_filter_row_f32(a.data(), ma.data(), W, r);
            box_filter_row_f32(b.data(), mb.data(), W, r);

            for (int i = 0; i < W; ++i) {
                float s = ma[i]*rf[i] + mb[i];
                if (mr[i]) { lsum[i] += rf[i] - s; ++lcnt[i]; }
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
    for (int x = 0; x < W; ++x)
        off[x] = std::max(-max_off, std::min(max_off, off[x]));

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const uint8_t* ir = img + y*W;
        float* dr = dst + y*W;
        for (int x = 0; x < W; ++x) dr[x] = (float)ir[x] - off[x];
    }
}

// ---------------------------------------------------------------------------
// Solution 3 — uint8_t row-streaming, combined weighted GF + accumulation
// ---------------------------------------------------------------------------
static void destripe_3(const uint8_t* img, float* dst,
                       int W, int H, int r, float eps, float eta,
                       float grad_thresh, int min_flat, float max_off) {
    std::vector<char> mask(W*H);
    compute_flat_mask_u8(img, mask.data(), W, H, grad_thresh);

    std::vector<float> off(W, 0.0f);
    std::vector<int> cnt(W, 0);
    std::vector<char> valid(W, 0);

    #pragma omp parallel
    {
        std::vector<float> rf(W), r2(W), mu(W), cr(W), a(W), b(W), ma(W), mb(W);
        std::vector<float> lsum(W, 0.0f);
        std::vector<int> lcnt(W, 0);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = img + y*W;
            const char* mr = mask.data() + y*W;

            for (int i = 0; i < W; ++i) { float v = row[i]; rf[i] = v; r2[i] = v*v; }
            box_filter_row_u8(row, mu.data(), W, r);
            box_filter_row_f32(r2.data(), cr.data(), W, r);

            float sv = 0;
            for (int i = 0; i < W; ++i) {
                float v = cr[i] - mu[i]*mu[i];
                r2[i] = v; sv += v;
            }
            float mv = sv / W;
            for (int i = 0; i < W; ++i) {
                float chi = (r2[i] + eta) / (mv + eta);
                a[i] = r2[i] / (r2[i] + eps / chi);
                b[i] = (1.0f - a[i]) * mu[i];
            }
            box_filter_row_f32(a.data(), ma.data(), W, r);
            box_filter_row_f32(b.data(), mb.data(), W, r);

            for (int i = 0; i < W; ++i) {
                float s = ma[i]*rf[i] + mb[i];
                if (mr[i]) { lsum[i] += rf[i] - s; ++lcnt[i]; }
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
    for (int x = 0; x < W; ++x)
        off[x] = std::max(-max_off, std::min(max_off, off[x]));

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
    for(int i=0;i<N;++i) destripe_1(img.data(),d1.data(),W,H,r,eps,mo);
    auto t1=std::chrono::high_resolution_clock::now();
    double s1=std::chrono::duration<double,std::milli>(t1-t0).count()/N;

    t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;++i) destripe_2(img.data(),d2.data(),W,H,r,eps,gt,mf,mo);
    t1=std::chrono::high_resolution_clock::now();
    double s2=std::chrono::duration<double,std::milli>(t1-t0).count()/N;

    t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;++i) destripe_3(img.data(),d3.data(),W,H,r,eps,eta,gt,mf,mo);
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
