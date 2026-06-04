#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>

inline void box_filter_row(const float* input, float* output, int W, int r) {
    float scale = 1.0f / (2 * r + 1);
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        sum += input[std::max(0, std::min(W - 1, i))];
    }
    output[0] = sum * scale;
    for (int x = 1; x < W; ++x) {
        int left = std::max(0, x - r - 1);
        int right = std::min(W - 1, x + r);
        sum += input[right] - input[left];
        output[x] = sum * scale;
    }
}

void guided_filter_horizontal(const float* __restrict__ img, float* __restrict__ out_smooth, int W, int H, int r, float eps) {
    #pragma omp parallel
    {
        std::vector<float> row_sq(W);
        std::vector<float> mu(W), corr(W), var(W);
        std::vector<float> a(W), b(W);
        std::vector<float> mean_a(W), mean_b(W);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const float* img_row = &img[y * W];
            float* smooth_row = &out_smooth[y * W];

            #pragma omp simd
            for (int x = 0; x < W; ++x) row_sq[x] = img_row[x] * img_row[x];

            box_filter_row(img_row, mu.data(), W, r);
            box_filter_row(row_sq.data(), corr.data(), W, r);

            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                float v = corr[x] - mu[x] * mu[x];
                var[x] = v;
                a[x] = v / (v + eps);
                b[x] = mu[x] - a[x] * mu[x];
            }

            box_filter_row(a.data(), mean_a.data(), W, r);
            box_filter_row(b.data(), mean_b.data(), W, r);

            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                smooth_row[x] = mean_a[x] * img_row[x] + mean_b[x];
            }
        }
    }
}

void weighted_guided_filter_horizontal(const float* __restrict__ img, float* __restrict__ out_smooth, int W, int H, int r, float eps, float eta) {
    #pragma omp parallel
    {
        std::vector<float> row_sq(W);
        std::vector<float> mu(W), corr(W), var(W);
        std::vector<float> a(W), b(W);
        std::vector<float> mean_a(W), mean_b(W);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const float* img_row = &img[y * W];
            float* smooth_row = &out_smooth[y * W];

            #pragma omp simd
            for (int x = 0; x < W; ++x) row_sq[x] = img_row[x] * img_row[x];

            box_filter_row(img_row, mu.data(), W, r);
            box_filter_row(row_sq.data(), corr.data(), W, r);

            float sum_var = 0.0f;
            #pragma omp simd reduction(+:sum_var)
            for (int x = 0; x < W; ++x) {
                float v = corr[x] - mu[x] * mu[x];
                var[x] = v;
                sum_var += v;
            }
            float mean_var = sum_var / W;

            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                float chi = (var[x] + eta) / (mean_var + eta);
                a[x] = var[x] / (var[x] + eps / chi);
                b[x] = mu[x] - a[x] * mu[x];
            }

            box_filter_row(a.data(), mean_a.data(), W, r);
            box_filter_row(b.data(), mean_b.data(), W, r);

            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                smooth_row[x] = mean_a[x] * img_row[x] + mean_b[x];
            }
        }
    }
}

void compute_flat_mask(const float* __restrict__ img, char* __restrict__ mask, int W, int H, float grad_thresh) {
    std::vector<float> temp(W * H);
    
    // Horizontal pass
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* src_row = img + y * W;
        float* temp_row = temp.data() + y * W;
        for (int x = 0; x < W; ++x) {
            int x_m2 = std::max(0, x - 2);
            int x_m1 = std::max(0, x - 1);
            int x_0  = x;
            int x_p1 = std::min(W - 1, x + 1);
            int x_p2 = std::min(W - 1, x + 2);
            temp_row[x] = (src_row[x_m2] + 4.0f * src_row[x_m1] + 6.0f * src_row[x_0] + 4.0f * src_row[x_p1] + src_row[x_p2]) * 0.0625f;
        }
    }
    
    std::vector<float> blurred(W * H);
    // Vertical pass
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* r_m2 = temp.data() + std::max(0, y - 2) * W;
        const float* r_m1 = temp.data() + std::max(0, y - 1) * W;
        const float* r_0  = temp.data() + y * W;
        const float* r_p1 = temp.data() + std::min(H - 1, y + 1) * W;
        const float* r_p2 = temp.data() + std::min(H - 1, y + 2) * W;
        float* dst_row = blurred.data() + y * W;

        #pragma omp simd
        for (int x = 0; x < W; ++x) {
            dst_row[x] = (r_m2[x] + 4.0f * r_m1[x] + 6.0f * r_0[x] + 4.0f * r_p1[x] + r_p2[x]) * 0.0625f;
        }
    }

    // Sobel gradients
    float sq_grad_thresh = grad_thresh * grad_thresh * 16.0f; // std::sqrt(gx*gx+gy*gy)/4 < th => gx*gx+gy*gy < th^2 * 16
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        char* mask_row = mask + y * W;
        if (y == 0 || y == H - 1) {
            for (int x = 0; x < W; ++x) mask_row[x] = 0;
            continue;
        }
        const float* r_m1 = blurred.data() + (y - 1) * W;
        const float* r_0  = blurred.data() + y * W;
        const float* r_p1 = blurred.data() + (y + 1) * W;

        mask_row[0] = 0;
        #pragma omp simd
        for (int x = 1; x < W - 1; ++x) {
            float gx = (r_m1[x+1] + 2.0f*r_0[x+1] + r_p1[x+1]) -
                       (r_m1[x-1] + 2.0f*r_0[x-1] + r_p1[x-1]);
            float gy = (r_p1[x-1] + 2.0f*r_p1[x] + r_p1[x+1]) -
                       (r_m1[x-1] + 2.0f*r_m1[x] + r_m1[x+1]);
            
            mask_row[x] = (gx*gx + gy*gy < sq_grad_thresh) ? 1 : 0;
        }
        mask_row[W - 1] = 0;
    }
}

void interpolate_offsets(float* offsets, const char* valid, int W) {
    int first_valid = -1;
    for (int x = 0; x < W; ++x) {
        if (valid[x]) {
            first_valid = x;
            break;
        }
    }
    if (first_valid == -1) {
        std::fill(offsets, offsets + W, 0.0f);
        return;
    }
    for (int x = 0; x < first_valid; ++x) {
        offsets[x] = offsets[first_valid];
    }
    int last_valid = first_valid;
    for (int x = first_valid + 1; x < W; ++x) {
        if (valid[x]) {
            float y0 = offsets[last_valid];
            float y1 = offsets[x];
            for (int k = last_valid + 1; k < x; ++k) {
                float t = static_cast<float>(k - last_valid) / (x - last_valid);
                offsets[k] = y0 + t * (y1 - y0);
            }
            last_valid = x;
        }
    }
    for (int x = last_valid + 1; x < W; ++x) {
        offsets[x] = offsets[last_valid];
    }
}

// ==========================================
// The 3 Solutions to Compare
// ==========================================

void destripe_solution_1(const float* __restrict__ img, float* __restrict__ dst, int W, int H, int r, float eps, float max_offset) {
    std::vector<float> smooth(W * H);
    guided_filter_horizontal(img, smooth.data(), W, H, r, eps);
    
    std::vector<float> col_sums(W, 0.0f);
    
    #pragma omp parallel
    {
        std::vector<float> local_sums(W, 0.0f);
        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const float* img_row = img + y * W;
            const float* smooth_row = smooth.data() + y * W;
            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                local_sums[x] += img_row[x] - smooth_row[x];
            }
        }
        #pragma omp critical
        {
            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                col_sums[x] += local_sums[x];
            }
        }
    }

    std::vector<float> offsets(W, 0.0f);
    float H_inv = 1.0f / H;
    for (int x = 0; x < W; ++x) {
        float avg = col_sums[x] * H_inv;
        offsets[x] = std::max(-max_offset, std::min(max_offset, avg));
    }
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* img_row = img + y * W;
        float* dst_row = dst + y * W;
        #pragma omp simd
        for (int x = 0; x < W; ++x) {
            dst_row[x] = img_row[x] - offsets[x];
        }
    }
}

void destripe_solution_2(const float* __restrict__ img, float* __restrict__ dst, int W, int H, int r, float eps, float grad_thresh, int min_flat, float max_offset) {
    std::vector<float> smooth(W * H);
    guided_filter_horizontal(img, smooth.data(), W, H, r, eps);
    
    std::vector<char> mask(W * H);
    compute_flat_mask(img, mask.data(), W, H, grad_thresh);
    
    std::vector<float> col_sums(W, 0.0f);
    std::vector<float> col_counts(W, 0.0f);
    
    #pragma omp parallel
    {
        std::vector<float> local_sums(W, 0.0f);
        std::vector<float> local_counts(W, 0.0f);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const float* img_row = img + y * W;
            const float* smooth_row = smooth.data() + y * W;
            const char* mask_row = mask.data() + y * W;
            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                if (mask_row[x]) {
                    local_sums[x] += img_row[x] - smooth_row[x];
                    local_counts[x] += 1.0f;
                }
            }
        }

        #pragma omp critical
        {
            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                col_sums[x] += local_sums[x];
                col_counts[x] += local_counts[x];
            }
        }
    }

    std::vector<float> offsets(W, 0.0f);
    std::vector<char> valid(W, 0);

    for (int x = 0; x < W; ++x) {
        if (col_counts[x] >= min_flat) {
            offsets[x] = col_sums[x] / col_counts[x];
            valid[x] = 1;
        }
    }
    
    interpolate_offsets(offsets.data(), valid.data(), W);
    
    for (int x = 0; x < W; ++x) {
        offsets[x] = std::max(-max_offset, std::min(max_offset, offsets[x]));
    }
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* img_row = img + y * W;
        float* dst_row = dst + y * W;
        #pragma omp simd
        for (int x = 0; x < W; ++x) {
            dst_row[x] = img_row[x] - offsets[x];
        }
    }
}

void destripe_solution_3(const float* __restrict__ img, float* __restrict__ dst, int W, int H, int r, float eps, float eta, float grad_thresh, int min_flat, float max_offset) {
    std::vector<float> smooth(W * H);
    weighted_guided_filter_horizontal(img, smooth.data(), W, H, r, eps, eta);
    
    std::vector<char> mask(W * H);
    compute_flat_mask(img, mask.data(), W, H, grad_thresh);
    
    std::vector<float> col_sums(W, 0.0f);
    std::vector<float> col_counts(W, 0.0f);
    
    #pragma omp parallel
    {
        std::vector<float> local_sums(W, 0.0f);
        std::vector<float> local_counts(W, 0.0f);

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const float* img_row = img + y * W;
            const float* smooth_row = smooth.data() + y * W;
            const char* mask_row = mask.data() + y * W;
            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                if (mask_row[x]) {
                    local_sums[x] += img_row[x] - smooth_row[x];
                    local_counts[x] += 1.0f;
                }
            }
        }

        #pragma omp critical
        {
            #pragma omp simd
            for (int x = 0; x < W; ++x) {
                col_sums[x] += local_sums[x];
                col_counts[x] += local_counts[x];
            }
        }
    }

    std::vector<float> offsets(W, 0.0f);
    std::vector<char> valid(W, 0);

    for (int x = 0; x < W; ++x) {
        if (col_counts[x] >= min_flat) {
            offsets[x] = col_sums[x] / col_counts[x];
            valid[x] = 1;
        }
    }
    
    interpolate_offsets(offsets.data(), valid.data(), W);
    
    for (int x = 0; x < W; ++x) {
        offsets[x] = std::max(-max_offset, std::min(max_offset, offsets[x]));
    }
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const float* img_row = img + y * W;
        float* dst_row = dst + y * W;
        #pragma omp simd
        for (int x = 0; x < W; ++x) {
            dst_row[x] = img_row[x] - offsets[x];
        }
    }
}

// ==========================================
// Main Benchmark Loader
// ==========================================

int main() {
    const int W = 2048;
    const int H = 2048;
    const int size = W * H;
    
    // Original image was uint8 but we've been reading float32. We can convert directly.
    std::vector<unsigned char> img_u8(size);
    std::ifstream in("hikrobot.raw_u8", std::ios::binary);
    if (!in) {
        std::cerr << "Falling back to float raw..." << std::endl;
        std::vector<float> img_f32(size);
        std::ifstream inf("hikrobot.raw", std::ios::binary);
        if (!inf) {
            std::cerr << "Could not open hikrobot.raw" << std::endl;
            return 1;
        }
        inf.read(reinterpret_cast<char*>(img_f32.data()), size * sizeof(float));
        inf.close();
        for(int i = 0; i < size; ++i) {
            img_u8[i] = static_cast<unsigned char>(std::clamp(img_f32[i], 0.0f, 255.0f));
        }
    } else {
        in.read(reinterpret_cast<char*>(img_u8.data()), size);
        in.close();
    }

    std::vector<float> img(size);
    for(int i = 0; i < size; ++i) img[i] = static_cast<float>(img_u8[i]);
    
    std::vector<float> dst1(size);
    std::vector<float> dst2(size);
    std::vector<float> dst3(size);
    
    const int r = 102; // filter_size = 205 (radius = 102)
    const float eps = 50.0f;
    const float eta = 10.0f;
    const float grad_thresh = 3.0f;
    const int min_flat = 10;
    const float max_offset = 3.0f;
    
    // Warmup OpenMP
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
    }
    
    const int iterations = 10;
    
    std::cout << "Benchmarking Optimized C++ implementations (averaged over " << iterations << " runs, W=" << W << ", H=" << H << ")..." << std::endl;
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads." << std::endl;
    
    // Benchmark Solution 1 (Sui Original)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        destripe_solution_1(img.data(), dst1.data(), W, H, r, eps, max_offset);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_sui = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
    std::cout << "Solution 1 (Sui Original): " << time_sui << " ms" << std::endl;
    
    // Benchmark Solution 2 (Masked Guided Filter)
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        destripe_solution_2(img.data(), dst2.data(), W, H, r, eps, grad_thresh, min_flat, max_offset);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double time_masked = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
    std::cout << "Solution 2 (Masked GF): " << time_masked << " ms" << std::endl;
    
    // Benchmark Solution 3 (SOTA Masked Weighted Guided Filter)
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        destripe_solution_3(img.data(), dst3.data(), W, H, r, eps, eta, grad_thresh, min_flat, max_offset);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double time_weighted = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
    std::cout << "Solution 3 (Masked Weighted GF): " << time_weighted << " ms" << std::endl;
    
    // Save output binary files for post-processing PNG check
    std::ofstream out1("hikrobot_cpp_sui.raw", std::ios::binary);
    out1.write(reinterpret_cast<char*>(dst1.data()), size * sizeof(float));
    out1.close();
    
    std::ofstream out2("hikrobot_cpp_masked.raw", std::ios::binary);
    out2.write(reinterpret_cast<char*>(dst2.data()), size * sizeof(float));
    out2.close();
    
    std::ofstream out3("hikrobot_cpp_weighted.raw", std::ios::binary);
    out3.write(reinterpret_cast<char*>(dst3.data()), size * sizeof(float));
    out3.close();
    
    std::cout << "All outputs written successfully." << std::endl;
    return 0;
}
