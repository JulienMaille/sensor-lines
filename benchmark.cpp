#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>

// Horizontal sliding window box filter: O(1) per pixel
void box_filter_row(const float* input, float* output, int W, int r) {
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

// 1. Standard 1D Guided Filter (Sui Original)
void guided_filter_horizontal(const float* img, float* out_smooth, int W, int H, int r, float eps) {
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

            for (int x = 0; x < W; ++x) row_sq[x] = img_row[x] * img_row[x];

            box_filter_row(img_row, mu.data(), W, r);
            box_filter_row(row_sq.data(), corr.data(), W, r);

            for (int x = 0; x < W; ++x) {
                var[x] = corr[x] - mu[x] * mu[x];
                a[x] = var[x] / (var[x] + eps);
                b[x] = mu[x] - a[x] * mu[x];
            }

            box_filter_row(a.data(), mean_a.data(), W, r);
            box_filter_row(b.data(), mean_b.data(), W, r);

            for (int x = 0; x < W; ++x) {
                smooth_row[x] = mean_a[x] * img_row[x] + mean_b[x];
            }
        }
    }
}

// 2. 1D Horizontal Weighted Guided Filter (WGIF)
void weighted_guided_filter_horizontal(const float* img, float* out_smooth, int W, int H, int r, float eps, float eta) {
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

            for (int x = 0; x < W; ++x) row_sq[x] = img_row[x] * img_row[x];

            box_filter_row(img_row, mu.data(), W, r);
            box_filter_row(row_sq.data(), corr.data(), W, r);

            float sum_var = 0.0f;
            for (int x = 0; x < W; ++x) {
                var[x] = corr[x] - mu[x] * mu[x];
                sum_var += var[x];
            }
            float mean_var = sum_var / W;

            for (int x = 0; x < W; ++x) {
                float chi = (var[x] + eta) / (mean_var + eta);
                a[x] = var[x] / (var[x] + eps / chi);
                b[x] = mu[x] - a[x] * mu[x];
            }

            box_filter_row(a.data(), mean_a.data(), W, r);
            box_filter_row(b.data(), mean_b.data(), W, r);

            for (int x = 0; x < W; ++x) {
                smooth_row[x] = mean_a[x] * img_row[x] + mean_b[x];
            }
        }
    }
}

// Separable 5x5 Gaussian blur: [1, 4, 6, 4, 1] / 16
void gaussian_blur_5x5(const float* src, float* dst, int W, int H) {
    std::vector<float> temp(W * H);
    
    // Horizontal pass
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0;
            for (int k = -2; k <= 2; ++k) {
                int xx = std::max(0, std::min(W - 1, x + k));
                float weight = (k == -2 || k == 2) ? 1.0f : ((k == -1 || k == 1) ? 4.0f : 6.0f);
                sum += src[y * W + xx] * weight;
            }
            temp[y * W + x] = sum / 16.0f;
        }
    }
    
    // Vertical pass
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0;
            for (int k = -2; k <= 2; ++k) {
                int yy = std::max(0, std::min(H - 1, y + k));
                float weight = (k == -2 || k == 2) ? 1.0f : ((k == -1 || k == 1) ? 4.0f : 6.0f);
                sum += temp[yy * W + x] * weight;
            }
            dst[y * W + x] = sum / 16.0f;
        }
    }
}

// Compute binary flat mask using Sobel gradients (stored as char to avoid std::vector<bool> specialization)
void compute_flat_mask(const float* img, char* mask, int W, int H, float grad_thresh) {
    std::vector<float> blurred(W * H);
    gaussian_blur_5x5(img, blurred.data(), W, H);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (y == 0 || y == H - 1 || x == 0 || x == W - 1) {
                mask[y * W + x] = 0;
                continue;
            }
            float gx = (blurred[(y-1)*W + x+1] + 2.0f*blurred[y*W + x+1] + blurred[(y+1)*W + x+1]) -
                       (blurred[(y-1)*W + x-1] + 2.0f*blurred[y*W + x-1] + blurred[(y+1)*W + x-1]);
            float gy = (blurred[(y+1)*W + x-1] + 2.0f*blurred[(y+1)*W + x] + blurred[(y+1)*W + x+1]) -
                       (blurred[(y-1)*W + x-1] + 2.0f*blurred[(y-1)*W + x] + blurred[(y-1)*W + x+1]);
            
            float grad_mag = std::sqrt(gx*gx + gy*gy) / 4.0f;
            mask[y * W + x] = (grad_mag < grad_thresh) ? 1 : 0;
        }
    }
}

// 1D Linear Interpolation of invalid column offsets (using char array for validation flags)
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

// Solution 1: Sui Original
void destripe_solution_1(const float* img, float* dst, int W, int H, int r, float eps, float max_offset) {
    std::vector<float> smooth(W * H);
    guided_filter_horizontal(img, smooth.data(), W, H, r, eps);
    
    std::vector<float> offsets(W, 0.0f);
    
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < W; ++x) {
        float sum = 0.0f;
        for (int y = 0; y < H; ++y) {
            sum += img[y * W + x] - smooth[y * W + x];
        }
        float avg = sum / H;
        offsets[x] = std::max(-max_offset, std::min(max_offset, avg));
    }
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            dst[y * W + x] = img[y * W + x] - offsets[x];
        }
    }
}

// Solution 2: Masked Guided Filter (Algo F)
void destripe_solution_2(const float* img, float* dst, int W, int H, int r, float eps, float grad_thresh, int min_flat, float max_offset) {
    std::vector<float> smooth(W * H);
    guided_filter_horizontal(img, smooth.data(), W, H, r, eps);
    
    std::vector<char> mask(W * H);
    compute_flat_mask(img, mask.data(), W, H, grad_thresh);
    
    std::vector<float> offsets(W, 0.0f);
    std::vector<char> valid(W, 0);
    
    #pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < W; ++x) {
        float sum = 0.0f;
        float count = 0.0f;
        for (int y = 0; y < H; ++y) {
            int idx = y * W + x;
            if (mask[idx]) {
                sum += img[idx] - smooth[idx];
                count += 1.0f;
            }
        }
        if (count >= min_flat) {
            offsets[x] = sum / count;
            valid[x] = 1;
        }
    }
    
    interpolate_offsets(offsets.data(), valid.data(), W);
    
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < W; ++x) {
        offsets[x] = std::max(-max_offset, std::min(max_offset, offsets[x]));
    }
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            dst[y * W + x] = img[y * W + x] - offsets[x];
        }
    }
}

// Solution 3: Masked Weighted Guided Filter (SOTA - Algo G)
void destripe_solution_3(const float* img, float* dst, int W, int H, int r, float eps, float eta, float grad_thresh, int min_flat, float max_offset) {
    std::vector<float> smooth(W * H);
    weighted_guided_filter_horizontal(img, smooth.data(), W, H, r, eps, eta);
    
    std::vector<char> mask(W * H);
    compute_flat_mask(img, mask.data(), W, H, grad_thresh);
    
    std::vector<float> offsets(W, 0.0f);
    std::vector<char> valid(W, 0);
    
    #pragma omp parallel for schedule(dynamic)
    for (int x = 0; x < W; ++x) {
        float sum = 0.0f;
        float count = 0.0f;
        for (int y = 0; y < H; ++y) {
            int idx = y * W + x;
            if (mask[idx]) {
                sum += img[idx] - smooth[idx];
                count += 1.0f;
            }
        }
        if (count >= min_flat) {
            offsets[x] = sum / count;
            valid[x] = 1;
        }
    }
    
    interpolate_offsets(offsets.data(), valid.data(), W);
    
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < W; ++x) {
        offsets[x] = std::max(-max_offset, std::min(max_offset, offsets[x]));
    }
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            dst[y * W + x] = img[y * W + x] - offsets[x];
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
    
    std::vector<float> img(size);
    std::ifstream in("hikrobot.raw", std::ios::binary);
    if (!in) {
        std::cerr << "Could not open hikrobot.raw" << std::endl;
        return 1;
    }
    in.read(reinterpret_cast<char*>(img.data()), size * sizeof(float));
    in.close();
    
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
    
    std::cout << "Benchmarking C++ implementations (averaged over " << iterations << " runs, W=" << W << ", H=" << H << ")..." << std::endl;
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
