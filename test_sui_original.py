import cv2
import numpy as np
import scipy.ndimage
import time

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

def guided_filter_horizontal_constant_eps(img, r, eps):
    """
    Standard 1D Horizontal Guided Filter (constant eps, no weighting)
    As described in Sui et al. (2012)
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

def destripe_sui_original(img, filter_size=101, eps=50.0, max_offset=3.0):
    """
    Original Sui et al. (2012) Destriping Algorithm
    - 1D Horizontal Guided Filter (constant eps)
    - Column averages computed over all pixels (no flat-region mask)
    - Offsets clipped to max_offset
    """
    start_time = time.perf_counter()
    H, W = img.shape
    r = filter_size // 2
    
    # 1. 1D Horizontal Guided Filter
    img_smooth_rows = guided_filter_horizontal_constant_eps(img, r, eps)
    
    # 2. High-pass filter along each row
    img_hp = img - img_smooth_rows
    
    # 3. Global column average over all rows (no flat-region masking)
    offsets = np.mean(img_hp, axis=0)
    
    # 4. Limit corrections to the specified maximum offset
    if max_offset is not None:
        offsets = np.clip(offsets, -max_offset, max_offset)
        
    # 5. Subtract offsets from each row
    corrected = img - offsets[np.newaxis, :]
    
    elapsed = (time.perf_counter() - start_time) * 1000.0
    return corrected, offsets, elapsed

def main():
    img_path = "hikrobot.png"
    print(f"Loading image: {img_path}...")
    img = load_image(img_path)
    H, W = img.shape
    print(f"Dimensions: {W}x{H}")
    
    filter_size = (W // 10) | 1
    print(f"\nRunning original Sui et al. (2012) algorithm...")
    
    corr_sui, offsets_sui, t_sui = destripe_sui_original(img, filter_size=filter_size, eps=50.0, max_offset=3.0)
    print(f"Original Sui Algorithm: {t_sui:.2f} ms")
    
    # Evaluate noise levels in flat ROI
    y_start, y_end = int(H * 0.35), int(H * 0.55)
    x_start, x_end = int(W * 0.1), int(W * 0.9)
    
    roi = corr_sui[y_start:y_end, x_start:x_end]
    col_means = np.mean(roi, axis=0)
    col_means_smooth = scipy.ndimage.median_filter(col_means, size=31)
    noise = col_means - col_means_smooth
    std_noise = np.std(noise)
    print(f"ROI Column-to-Column Noise StdDev (Original Sui): {std_noise:.4f}")
    
    # Save the output images
    print("\nSaving output images...")
    save_image(corr_sui, "hikrobot_sui_original.png")
    save_image(apply_gamma(corr_sui, 2.5), "hikrobot_sui_original_gamma_enhanced.png")
    print("Done! Outputs saved to hikrobot_sui_original.png and hikrobot_sui_original_gamma_enhanced.png")

if __name__ == "__main__":
    main()
