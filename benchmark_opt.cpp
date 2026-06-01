#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <chrono>
#include <omp.h>

// ---------------------------------------------------------------------------
// Box filter with 3-way loop split for hot path (no min/max in middle).
// Initial sum matches original std::max/min border-replicate exactly:
//   sum = sum_{i=-r}^{r} in[clamp(i, 0, W-1)]
// ---------------------------------------------------------------------------
static void box_filter_row(const float* in, float* out, int W, int r) {
    const float s = 1.0f / (2 * r + 1);
    if (W <= 0) return;

    // initial window sum — match original border replication exactly
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i)
        sum += in[std::max(0, std::min(W - 1, i))];
    out[0] = sum * s;

    int x = 1;
    // leading edge: left clamped to 0
    int le = (r + 1 < W) ? (r + 1) : W;
    for (; x < le; ++x) {
        sum += in[x + r] - in[0];
        out[x] = sum * s;
    }
    // middle: no clamping
    int me = (W - r > le) ? (W - r) : le;
    for (; x < me; ++x) {
        sum += in[x + r] - in[x - r - 1];
        out[x] = sum * s;
    }
    // trailing edge: right clamped to W-1
    for (; x < W; ++x) {
        sum += in[W - 1] - in[x - r - 1];
        out[x] = sum * s;
    }
}

// ---------------------------------------------------------------------------
// 1D Horizontal Guided Filter
// ---------------------------------------------------------------------------
static void guided_filter(const float* img, float* out,
                          int W, int H, int r, float eps) {
    #pragma omp parallel
    {
        std::vector<float> r2(W), mu(W), cr(W), a(W), b(W), ma(W), mb(W);
        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const float* row = img + y * W;
            float* sr = out + y * W;
            for (int i = 0; i < W; ++i) r2[i] = row[i] * row[i];
            box_filter_row(row, mu.data(), W, r);
            box_filter_row(r2.data(), cr.data(), W, r);
            for (int i = 0; i < W; ++i) {
                float v = cr[i] - mu[i] * mu[i];
                a[i] = v / (v + eps);
                b[i] = (1.0f - a[i]) * mu[i];
            }
            box_filter_row(a.data(), ma.data(), W, r);
            box_filter_row(b.data(), mb.data(), W, r);
            for (int i = 0; i < W; ++i) sr[i] = ma[i] * row[i] + mb[i];
        }
    }
}

// ---------------------------------------------------------------------------
// 1D Horizontal Weighted Guided Filter
// ---------------------------------------------------------------------------
static void weighted_guided_filter(const float* img, float* out,
                                   int W, int H, int r, float eps, float eta) {
    #pragma omp parallel
    {
        std::vector<float> t(W), mu(W), cr(W), a(W), b(W), ma(W), mb(W);
        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const float* row = img + y * W;
            float* sr = out + y * W;
            for (int i = 0; i < W; ++i) t[i] = row[i] * row[i];
            box_filter_row(row, mu.data(), W, r);
            box_filter_row(t.data(), cr.data(), W, r);
            float sv = 0.0f;
            for (int i = 0; i < W; ++i) {
                float v = cr[i] - mu[i] * mu[i];
                t[i] = v; sv += v;
            }
            float mv = sv / W;
            for (int i = 0; i < W; ++i) {
                float chi = (t[i] + eta) / (mv + eta);
                a[i] = t[i] / (t[i] + eps / chi);
                b[i] = (1.0f - a[i]) * mu[i];
            }
            box_filter_row(a.data(), ma.data(), W, r);
            box_filter_row(b.data(), mb.data(), W, r);
            for (int i = 0; i < W; ++i) sr[i] = ma[i] * row[i] + mb[i];
        }
    }
}

// ---------------------------------------------------------------------------
// 5x5 separable Gaussian [1 4 6 4 1] / 16 — unrolled, loop-split for
// border handling, avoid min/max in hot path.
// ---------------------------------------------------------------------------
static void gaussian_5x5(const float* src, float* dst, int W, int H) {
    std::vector<float> tmp(W * H);

    // Horizontal pass
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* r = src + y * W;
        float* t = tmp.data() + y * W;

        for (int x = 0; x < 2 && x < W; ++x) {
            int xp1 = (x+1 < W) ? x+1 : W-1;
            int xp2 = (x+2 < W) ? x+2 : W-1;
            t[x] = (r[0] + 4*r[0] + 6*r[x] + 4*r[xp1] + r[xp2]) * (1.0f/16.0f);
        }
        for (int x = 2; x < W-2; ++x)
            t[x] = (r[x-2] + 4*r[x-1] + 6*r[x] + 4*r[x+1] + r[x+2]) * (1.0f/16.0f);
        for (int x = (W-2 > 2 ? W-2 : 2); x < W; ++x) {
            int xm2 = (x-2 >= 0) ? x-2 : 0;
            int xm1 = (x-1 >= 0) ? x-1 : 0;
            int xp1 = (x+1 < W) ? x+1 : W-1;
            int xp2 = (x+2 < W) ? x+2 : W-1;
            t[x] = (r[xm2] + 4*r[xm1] + 6*r[x] + 4*r[xp1] + r[xp2]) * (1.0f/16.0f);
        }
    }

    // Vertical pass  — loop-split: top border, main, bottom border
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        float* d = dst + y * W;
        const float* t = tmp.data();
        if (y < 2) {
            int ym2 = 0, ym1 = 0, yp1 = (y+1<H?y+1:H-1), yp2 = (y+2<H?y+2:H-1);
            for (int i = 0; i < W; ++i)
                d[i] = (t[ym2*W+i] + 4*t[ym1*W+i] + 6*t[y*W+i]
                      + 4*t[yp1*W+i] + t[yp2*W+i]) / 16.0f;
        } else if (y >= H-2) {
            int ym2 = y-2, ym1 = y-1, yp1 = H-1, yp2 = H-1;
            for (int i = 0; i < W; ++i)
                d[i] = (t[ym2*W+i] + 4*t[ym1*W+i] + 6*t[y*W+i]
                      + 4*t[yp1*W+i] + t[yp2*W+i]) / 16.0f;
        } else {
            const float* t0 = t + (y-2)*W, *t1 = t + (y-1)*W;
            const float* t2 = t + y*W,     *t3 = t + (y+1)*W;
            const float* t4 = t + (y+2)*W;
            for (int i = 0; i < W; ++i)
                d[i] = (t0[i] + 4*t1[i] + 6*t2[i] + 4*t3[i] + t4[i]) / 16.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Flat mask via Sobel on Gaussian-blurred image.
// Sobel loop-split: border handled separately.
// ---------------------------------------------------------------------------
static void compute_flat_mask(const float* img, char* mask,
                              int W, int H, float grad_thresh) {
    std::vector<float> bl(W * H);
    gaussian_5x5(img, bl.data(), W, H);

    if (H > 0) {
        std::fill(mask, mask + W, 0);
        std::fill(mask + (H-1)*W, mask + H*W, 0);
    }

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
            for (int k = lv+1; k < x; ++k)
                off[k] = y0 + (k-lv)*id*(y1-y0);
            lv = x;
        }
    }
    for (int x = lv+1; x < W; ++x) off[x] = off[lv];
}

// ---------------------------------------------------------------------------
// Solution 1
// ---------------------------------------------------------------------------
static void destripe_1(const float* img, float* dst,
                       int W, int H, int r, float eps, float max_off) {
    std::vector<float> smooth(W*H);
    guided_filter(img, smooth.data(), W, H, r, eps);

    std::vector<float> off(W, 0.0f);
    #pragma omp parallel
    {
        std::vector<float> loc(W, 0.0f);
        #pragma omp for schedule(static)
        for (int y0 = 0; y0 < H; y0 += 64) {
            int y1 = (y0+64 < H) ? y0+64 : H;
            for (int y = y0; y < y1; ++y) {
                const float* ir = img + y*W;
                const float* sr = smooth.data() + y*W;
                for (int x = 0; x < W; ++x) loc[x] += ir[x] - sr[x];
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
        const float* ir = img + y*W;
        float* dr = dst + y*W;
        for (int x = 0; x < W; ++x) dr[x] = ir[x] - off[x];
    }
}

// ---------------------------------------------------------------------------
// Solution 2
// ---------------------------------------------------------------------------
static void destripe_2(const float* img, float* dst,
                       int W, int H, int r, float eps,
                       float grad_thresh, int min_flat, float max_off) {
    std::vector<float> smooth(W*H);
    guided_filter(img, smooth.data(), W, H, r, eps);

    std::vector<char> mask(W*H);
    compute_flat_mask(img, mask.data(), W, H, grad_thresh);

    std::vector<float> off(W, 0.0f);
    std::vector<int> cnt(W, 0);
    std::vector<char> valid(W, 0);

    #pragma omp parallel
    {
        std::vector<float> lsum(W, 0.0f);
        std::vector<int> lcnt(W, 0);
        #pragma omp for schedule(static)
        for (int y0 = 0; y0 < H; y0 += 64) {
            int y1 = (y0+64 < H) ? y0+64 : H;
            for (int y = y0; y < y1; ++y) {
                const float* ir = img + y*W;
                const float* sr = smooth.data() + y*W;
                const char* mr = mask.data() + y*W;
                for (int x = 0; x < W; ++x) {
                    if (mr[x]) { lsum[x] += ir[x] - sr[x]; ++lcnt[x]; }
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
    for (int x = 0; x < W; ++x)
        off[x] = std::max(-max_off, std::min(max_off, off[x]));

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* ir = img + y*W;
        float* dr = dst + y*W;
        for (int x = 0; x < W; ++x) dr[x] = ir[x] - off[x];
    }
}

// ---------------------------------------------------------------------------
// Solution 3
// ---------------------------------------------------------------------------
static void destripe_3(const float* img, float* dst,
                       int W, int H, int r, float eps, float eta,
                       float grad_thresh, int min_flat, float max_off) {
    std::vector<float> smooth(W*H);
    weighted_guided_filter(img, smooth.data(), W, H, r, eps, eta);

    std::vector<char> mask(W*H);
    compute_flat_mask(img, mask.data(), W, H, grad_thresh);

    std::vector<float> off(W, 0.0f);
    std::vector<int> cnt(W, 0);
    std::vector<char> valid(W, 0);

    #pragma omp parallel
    {
        std::vector<float> lsum(W, 0.0f);
        std::vector<int> lcnt(W, 0);
        #pragma omp for schedule(static)
        for (int y0 = 0; y0 < H; y0 += 64) {
            int y1 = (y0+64 < H) ? y0+64 : H;
            for (int y = y0; y < y1; ++y) {
                const float* ir = img + y*W;
                const float* sr = smooth.data() + y*W;
                const char* mr = mask.data() + y*W;
                for (int x = 0; x < W; ++x) {
                    if (mr[x]) { lsum[x] += ir[x] - sr[x]; ++lcnt[x]; }
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
    for (int x = 0; x < W; ++x)
        off[x] = std::max(-max_off, std::min(max_off, off[x]));

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* ir = img + y*W;
        float* dr = dst + y*W;
        for (int x = 0; x < W; ++x) dr[x] = ir[x] - off[x];
    }
}

// ===================================================================
int main() {
    const int W=2048, H=2048, sz=W*H;
    std::vector<float> img(sz);
    std::ifstream("hikrobot.raw",std::ios::binary)
        .read((char*)img.data(),sz*4);
    std::vector<float> d1(sz),d2(sz),d3(sz);

    const int r=102; const float eps=50,eta=10,gt=3,mo=3; const int mf=10;

    #pragma omp parallel
    { volatile int _ = omp_get_thread_num(); (void)_; }
    std::cout<<omp_get_max_threads()<<" threads\n";

    const int N=10;
    std::cout<<"Benchmark (avg "<<N<<", "<<W<<"x"<<H<<")\n\n";

    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;++i)destripe_1(img.data(),d1.data(),W,H,r,eps,mo);
    auto t1=std::chrono::high_resolution_clock::now();
    double s1=std::chrono::duration<double,std::milli>(t1-t0).count()/N;

    t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;++i)destripe_2(img.data(),d2.data(),W,H,r,eps,gt,mf,mo);
    t1=std::chrono::high_resolution_clock::now();
    double s2=std::chrono::duration<double,std::milli>(t1-t0).count()/N;

    t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;++i)destripe_3(img.data(),d3.data(),W,H,r,eps,eta,gt,mf,mo);
    t1=std::chrono::high_resolution_clock::now();
    double s3=std::chrono::duration<double,std::milli>(t1-t0).count()/N;

    double b1=38.44,b2=61.82,b3=84.30;
    std::cout<<"========= RESULTS =========\n"
             <<"S1 (Sui Orig):       "<<s1<<" ms  ("<<(b1/s1)<<"x)\n"
             <<"S2 (Masked GF):      "<<s2<<" ms  ("<<(b2/s2)<<"x)\n"
             <<"S3 (Masked WGF):     "<<s3<<" ms  ("<<(b3/s3)<<"x)\n";

    std::ofstream("s1_opt.raw",std::ios::binary).write((char*)d1.data(),sz*4);
    std::ofstream("s2_opt.raw",std::ios::binary).write((char*)d2.data(),sz*4);
    std::ofstream("s3_opt.raw",std::ios::binary).write((char*)d3.data(),sz*4);
    std::cout<<"\nDone.\n";
    return 0;
}
