import cv2
import numpy as np
import scipy.ndimage
import time
import os
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img.astype(np.float32)

def save_image(img, path):
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)

def apply_gamma(img, gamma=2.5):
    img_norm = np.clip(img / 255.0, 0, 1)
    img_gamma = np.power(img_norm, gamma) * 255.0
    return img_gamma

def invert_gamma(img_gamma, gamma=2.5):
    img_norm = np.clip(img_gamma / 255.0, 0, 1)
    img_linear = np.power(img_norm, 1.0 / gamma) * 255.0
    return img_linear

# ==========================================
# SOTA Guided Filter and Weighted Guided Filter (1D)
# ==========================================

def guided_filter_horizontal(img, r, eps):
    """
    1D Horizontal Guided Filter on a 2D image
    """
    def box_filter(x):
        return cv2.blur(x, (2*r+1, 1), borderType=cv2.BORDER_REPLICATE)
        
    mu = box_filter(img)
    corr = box_filter(img * img)
    var = corr - mu * mu
    
    a = var / (var + eps)
    b = mu - a * mu
    
    mean_a = box_filter(a)
    mean_b = box_filter(b)
    
    q = mean_a * img + mean_b
    return q

def weighted_guided_filter_horizontal(img, r, eps, eta=10.0):
    """
    1D Horizontal Weighted Guided Filter (WGIF) on a 2D image
    Dynamically adjusts regularizer based on local 1D variance along the row.
    """
    def box_filter(x):
        return cv2.blur(x, (2*r+1, 1), borderType=cv2.BORDER_REPLICATE)
        
    mu = box_filter(img)
    corr = box_filter(img * img)
    var = corr - mu * mu
    
    # Calculate local edge weights
    mean_var = np.mean(var)
    chi = (var + eta) / (mean_var + eta)
    
    # Adapt eps
    a = var / (var + eps / chi)
    b = mu - a * mu
    
    mean_a = box_filter(a)
    mean_b = box_filter(b)
    
    q = mean_a * img + mean_b
    return q

# ==========================================
# Main Destriping Framework
# ==========================================

def destripe_algo_f(img, filter_size=101, eps=50.0, grad_thresh=3.0, min_flat_pixels=10):
    """
    Algorithm F: Masked Horizontal Guided Filter
    """
    start_time = time.perf_counter()
    H, W = img.shape
    r = filter_size // 2
    
    img_smooth_rows = guided_filter_horizontal(img, r, eps)
    img_hp = img - img_smooth_rows
    
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    grad_x = np.abs(cv2.Sobel(img_blurred, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(img_blurred, cv2.CV_32F, 0, 1, ksize=3))
    flat_mask = (grad_x < grad_thresh) & (grad_y < grad_thresh)
    
    weights = flat_mask.astype(np.float32)
    col_weight_sum = np.sum(weights, axis=0)
    col_val_sum = np.sum(img_hp * weights, axis=0)
    
    offsets = np.zeros(W, dtype=np.float32)
    valid_cols = col_weight_sum >= min_flat_pixels
    offsets[valid_cols] = col_val_sum[valid_cols] / col_weight_sum[valid_cols]
    
    nans = ~valid_cols
    if np.any(nans):
        x_indices = np.arange(W)
        if np.all(nans):
            offsets = np.zeros(W, dtype=np.float32)
        else:
            offsets[nans] = np.interp(x_indices[nans], x_indices[~nans], offsets[~nans])
            
    corrected = img - offsets[np.newaxis, :]
    elapsed = (time.perf_counter() - start_time) * 1000.0
    return corrected, offsets, elapsed

def destripe_algo_g(img, filter_size=101, eps=50.0, eta=10.0, grad_thresh=3.0, min_flat_pixels=10, max_offset=3.0):
    """
    Algorithm G: Masked Horizontal Weighted Guided Filter (SOTA)
    """
    start_time = time.perf_counter()
    H, W = img.shape
    r = filter_size // 2
    
    # 1. Row smoothing using Weighted Guided Filter
    img_smooth_rows = weighted_guided_filter_horizontal(img, r, eps, eta)
    img_hp = img - img_smooth_rows
    
    # 2. Compute flat region mask
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    grad_x = np.abs(cv2.Sobel(img_blurred, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(img_blurred, cv2.CV_32F, 0, 1, ksize=3))
    flat_mask = (grad_x < grad_thresh) & (grad_y < grad_thresh)
    
    # 3. Vectorized mean of flat pixels along columns
    weights = flat_mask.astype(np.float32)
    col_weight_sum = np.sum(weights, axis=0)
    col_val_sum = np.sum(img_hp * weights, axis=0)
    
    offsets = np.zeros(W, dtype=np.float32)
    valid_cols = col_weight_sum >= min_flat_pixels
    offsets[valid_cols] = col_val_sum[valid_cols] / col_weight_sum[valid_cols]
    
    nans = ~valid_cols
    if np.any(nans):
        x_indices = np.arange(W)
        if np.all(nans):
            offsets = np.zeros(W, dtype=np.float32)
        else:
            offsets[nans] = np.interp(x_indices[nans], x_indices[~nans], offsets[~nans])
            
    # Apply hard clipping to guarantee no pixel is corrected by more than max_offset
    if max_offset is not None:
        offsets = np.clip(offsets, -max_offset, max_offset)
        # Alternatively, soft clipping: offsets = max_offset * np.tanh(offsets / max_offset)
            
    corrected = img - offsets[np.newaxis, :]
    elapsed = (time.perf_counter() - start_time) * 1000.0
    return corrected, offsets, elapsed

# ==========================================
# Testing & Main Routine
# ==========================================

def run_tests():
    # 1. Clean the main image: hikrobot.png
    print("Loading original image (hikrobot.png)...")
    img_orig = load_image("hikrobot.png")
    H, W = img_orig.shape
    filter_size = (W // 10) | 1
    
    print("\nRunning Algorithms on original image...")
    # Algo F
    corr_f, offsets_f, t_f = destripe_algo_f(img_orig, filter_size=filter_size, eps=50.0)
    print(f"Algo F (Guided Filter): {t_f:.2f} ms")
    
    # Algo G (SOTA)
    corr_g, offsets_g, t_g = destripe_algo_g(img_orig, filter_size=filter_size, eps=50.0, eta=10.0)
    print(f"Algo G (Weighted Guided): {t_g:.2f} ms")
    
    # 2. Clean the contrast-enhanced image (hikrobot-gamma.png)
    print("\nLoading gamma-contrast image (hikrobot-gamma.png)...")
    img_gamma = load_image("hikrobot-gamma.png")
    
    print("\nCleaning gamma image (Direct vs. Smart/Linear-Domain)...")
    # Method A: Direct application on the gamma image
    corr_gamma_direct, _, t_g_direct = destripe_algo_g(img_gamma, filter_size=filter_size, eps=50.0, eta=10.0)
    print(f"Direct correction on gamma image: {t_g_direct:.2f} ms")
    
    # Method B: Invert gamma first, correct in linear space, then re-apply gamma
    t_start = time.perf_counter()
    img_linear = invert_gamma(img_gamma, gamma=2.5)
    img_linear_corr, _, _ = destripe_algo_g(img_linear, filter_size=filter_size, eps=50.0, eta=10.0)
    corr_gamma_smart = apply_gamma(img_linear_corr, gamma=2.5)
    t_g_smart = (time.perf_counter() - t_start) * 1000.0
    print(f"Smart (Linear-domain) correction on gamma image: {t_g_smart:.2f} ms")
    
    # Evaluate noise levels in flat ROI
    y_start, y_end = int(H * 0.35), int(H * 0.55)
    x_start, x_end = int(W * 0.1), int(W * 0.9)
    
    def measure_noise(image_data, label):
        roi = image_data[y_start:y_end, x_start:x_end]
        col_means = np.mean(roi, axis=0)
        col_means_smooth = scipy.ndimage.median_filter(col_means, size=31)
        noise = col_means - col_means_smooth
        std_noise = np.std(noise)
        print(f"ROI Noise StdDev ({label}): {std_noise:.4f}")
        return std_noise, noise
        
    print("\nROI Noise levels for hikrobot.png:")
    std_orig, noise_orig = measure_noise(img_orig, "Original")
    std_f, noise_f = measure_noise(corr_f, "Algo F (Guided)")
    std_g, noise_g = measure_noise(corr_g, "Algo G (Weighted Guided)")
    
    print("\nROI Noise levels for hikrobot-gamma.png:")
    std_gamma_orig, noise_gamma_orig = measure_noise(img_gamma, "Gamma Original")
    std_gamma_direct, noise_gamma_direct = measure_noise(corr_gamma_direct, "Gamma Direct Corr")
    std_gamma_smart, noise_gamma_smart = measure_noise(corr_gamma_smart, "Gamma Smart Corr")
    
    # Save the output images
    print("\nSaving output images...")
    save_image(corr_f, "hikrobot_corrected_f.png")
    save_image(corr_g, "hikrobot_corrected_g.png")
    save_image(corr_gamma_direct, "hikrobot-gamma_corrected_direct.png")
    save_image(corr_gamma_smart, "hikrobot-gamma_corrected_smart.png")
    
    # Save gamma-enhanced versions of the corrected original image for easy validation
    save_image(apply_gamma(corr_f, 2.5), "hikrobot_corrected_f_gamma_enhanced.png")
    save_image(apply_gamma(corr_g, 2.5), "hikrobot_corrected_g_gamma_enhanced.png")
    
    # Generate profile comparison plot for original image
    plt.figure(figsize=(12, 8))
    x_coords = np.arange(x_start, x_end)
    
    plt.subplot(2, 1, 1)
    plt.plot(x_coords, noise_orig, label=f"Original (std={std_orig:.3f})", alpha=0.5, color='gray')
    plt.plot(x_coords, noise_f, label=f"Algo F: Guided (std={std_f:.3f})", alpha=0.7, color='blue')
    plt.plot(x_coords, noise_g, label=f"Algo G: SOTA Weighted Guided (std={std_g:.3f})", alpha=0.7, color='green')
    plt.title("hikrobot.png - High-Frequency Noise Profile")
    plt.xlabel("Column Index")
    plt.ylabel("Offset Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(2, 1, 2)
    plt.plot(x_coords, noise_gamma_orig, label=f"Gamma Original (std={std_gamma_orig:.3f})", alpha=0.5, color='gray')
    plt.plot(x_coords, noise_gamma_direct, label=f"Gamma Direct Corr (std={std_gamma_direct:.3f})", alpha=0.7, color='red')
    plt.plot(x_coords, noise_gamma_smart, label=f"Gamma Smart Corr (std={std_gamma_smart:.3f})", alpha=0.7, color='green')
    plt.title("hikrobot-gamma.png - High-Frequency Noise Profile")
    plt.xlabel("Column Index")
    plt.ylabel("Offset Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("sota_profiles_comparison.png", dpi=150)
    plt.close()
    
    print("\nProcessing complete! All files saved.")

if __name__ == "__main__":
    run_tests()
