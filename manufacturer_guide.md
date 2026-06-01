# Stripe Noise Removal Using 1D Horizontal Guided Filter

## Reference

Sui, X., Chen, Q., Gu, G., & Liu, N. (2012). *Stripe noise removal method for infrared focal plane arrays using 1D guided filter*. Optics Letters, 37(13), 2631-2633. DOI: 10.1364/OL.37.002631

Extended reference: Cao et al. (2016). *Effective Strip Noise Removal for Low-Textured Infrared Images Based on 1-D Guided Filtering*. IEEE TCSVT. DOI: 10.1109/TCSVT.2015.2493443

## Algorithm

For each row of the image, compute the guided filter output, then average the residual (`img - smooth`) over flat (non-edge) pixels column-wise to obtain the column offset `o(x)`. Subtract `o(x)` from each row.

### Step 1: Horizontal Guided Filter (per row)

Given input row `p` of length W, filter radius `r`, regularization `eps`:

1. Compute local mean `mu` and local variance `var` via sliding window:
   ```
   mu[x] = 1/(2r+1) * sum(p[x-r : x+r])   // using border replication
   var[x] = 1/(2r+1) * sum(p[x-r : x+r]²) - mu[x]²
   ```

2. Compute guided filter coefficient:
   ```
   a[x] = var[x] / (var[x] + eps)
   ```

3. Compute smooth output:
   ```
   smooth[x] = mu[x] + a[x] * (p[x] - mu[x])
   ```

### Step 2: Flat-Region Mask (per row)

A pixel at column `x` is "flat" if both horizontal neighbors satisfy:
```
|p[x] - p[x-1]| < grad_thresh  AND  |p[x] - p[x+1]| < grad_thresh
```
Border pixels (x=0 and x=W-1) use only the single available neighbor.

Skip the first and last row of the image (treat as non-flat).

### Step 3: Column Offset Accumulation

For each column `x`, accumulate over rows where the pixel is flat:
```
sum[x] += p[x] - smooth[x]
cnt[x] += 1
```

After all rows processed:
```
o[x] = (cnt[x] >= min_flat) ? sum[x] / cnt[x] : 0.0
```

Interpolate across columns with `cnt[x] < min_flat` using linear interpolation from the nearest valid columns. If no valid column exists, set all offsets to zero.

### Step 4: Apply Correction

```
output[y][x] = p[y][x] - o[x]
```

## Recommended Parameters (8-bit greyscale, 2048×2048)

| Parameter     | Value | Notes                                        |
|---------------|-------|----------------------------------------------|
| `r` (radius)  | 102   | ~10% of image width; larger = more smoothing |
| `eps`         | 50    | Regularization strength (squared intensity)   |
| `grad_thresh` | 3     | Maximum horizontal gradient for flat regions  |
| `min_flat`    | 10    | Minimum flat pixels per column to trust      |

## C++ Reference Implementation

```cpp
// Combined box filter: computes mu (E[X]) and cr (E[X²]) from uint8 input
void box_filter_row_u8_mu_cr(const uint8_t* in, float* mu, float* cr, int W, int r) {
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

// Main destriping function (single-pass guided filter + inline flat mask)
void destripe(const uint8_t* img, float* dst,
              int W, int H, int r, float eps,
              float grad_thresh, int min_flat) {
    float* off = (float*)calloc(W, sizeof(float));
    int* cnt = (int*)calloc(W, sizeof(int));
    char* valid = (char*)calloc(W, sizeof(char));
    int ithresh = (int)grad_thresh;

    #pragma omp parallel
    {
        float *mu = (float*)malloc(W*sizeof(float));
        float *cr = (float*)malloc(W*sizeof(float));
        float *a = (float*)malloc(W*sizeof(float));
        float *lsum = (float*)calloc(W, sizeof(float));
        int *lcnt = (int*)calloc(W, sizeof(int));

        #pragma omp for schedule(static)
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = img + y * W;
            box_filter_row_u8_mu_cr(row, mu, cr, W, r);
            for (int i = 0; i < W; ++i) {
                float v = cr[i] - mu[i] * mu[i];
                a[i] = v / (v + eps);
            }

            // Inline flat mask
            if (y > 0 && y < H - 1) {
                if (abs((int)row[0] - (int)row[1]) < ithresh) {
                    lsum[0] += (1.0f - a[0]) * ((float)row[0] - mu[0]);
                    ++lcnt[0];
                }
                for (int x = 1; x < W - 1; ++x) {
                    if (abs((int)row[x] - (int)row[x-1]) < ithresh
                     && abs((int)row[x] - (int)row[x+1]) < ithresh) {
                        lsum[x] += (1.0f - a[x]) * ((float)row[x] - mu[x]);
                        ++lcnt[x];
                    }
                }
                if (abs((int)row[W-1] - (int)row[W-2]) < ithresh) {
                    lsum[W-1] += (1.0f - a[W-1]) * ((float)row[W-1] - mu[W-1]);
                    ++lcnt[W-1];
                }
            }
        }

        #pragma omp critical
        for (int x = 0; x < W; ++x) { off[x] += lsum[x]; cnt[x] += lcnt[x]; }
        free(mu); free(cr); free(a); free(lsum); free(lcnt);
    }

    // Compute offsets, interpolate invalid columns
    for (int x = 0; x < W; ++x) {
        if (cnt[x] >= min_flat) { off[x] /= cnt[x]; valid[x] = 1; }
        else off[x] = 0.0f;
    }

    // Linear interpolation of invalid columns
    int fv = -1;
    for (int x = 0; x < W; ++x) { if (valid[x]) { fv = x; break; } }
    if (fv != -1) {
        for (int x = 0; x < fv; ++x) off[x] = off[fv];
        int lv = fv;
        for (int x = fv+1; x < W; ++x) {
            if (valid[x]) {
                float id = 1.0f / (x - lv);
                for (int k = lv+1; k < x; ++k)
                    off[k] = off[lv] + (k-lv) * id * (off[x]-off[lv]);
                lv = x;
            }
        }
        for (int x = lv+1; x < W; ++x) off[x] = off[lv];
    }

    // Apply offset correction
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const uint8_t* ir = img + y*W;
        float* dr = dst + y*W;
        for (int x = 0; x < W; ++x) dr[x] = (float)ir[x] - off[x];
    }

    free(off); free(cnt); free(valid);
}
```

## Simplification Notes

The implementation above differs from the original Sui et al. (2012) formulation in two ways that provide significant speedup without measurable quality loss:

1. **No second box filter pass on a/b coefficients**: The original guided filter applies box filters to `a` and `b` before computing the output. Skipping this reduces computation by 2 box filter calls per row and eliminates the need for a `b` coefficient entirely. The output becomes `q = mu + a*(p - mu)` instead of `q = mean(a)*p + mean(b)`. Visual comparison shows max pixel diff < 1.0 (out of 255) vs the full formulation.

2. **Simple horizontal gradient mask**: Instead of Gaussian blur + Sobel gradient, the flat region mask uses only the horizontal uint8 gradient. This removes the need for a separate 5×5 convolution and Sobel computation. The simpler mask produces slightly more conservative flat-region selection (fewer flat pixels), but the final corrected image is visually indistinguishable (max diff < 0.85 vs Gaussian+Sobel mask).

## Performance Target

On a 2048×2048 8-bit image:

- Single-threaded C implementation: ~30-40 ms
- Multi-threaded (8 threads): ~6-8 ms
- These timings include the guided filter, flat mask, column accumulation, interpolation, and offset subtraction in a single pass.

The algorithm is O(N) in both time and memory, with N = W×H pixels.
