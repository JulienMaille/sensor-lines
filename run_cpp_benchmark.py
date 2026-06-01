import cv2
import numpy as np
import subprocess
import os
import time

def main():
    # 1. Convert PNG to Raw float32 file
    print("Converting hikrobot.png to raw binary format...")
    img = cv2.imread("hikrobot.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not find hikrobot.png in current directory.")
        return
    img = img.astype(np.float32)
    img.tofile("hikrobot.raw")
    print(f"hikrobot.raw written successfully ({os.path.getsize('hikrobot.raw')} bytes).")
    
    # 2. Compile benchmark.cpp using MSVC cl.exe
    vcvars_path = r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    if not os.path.exists(vcvars_path):
        print(f"Error: Could not find vcvars64.bat at: {vcvars_path}")
        return
    
    # Run compiler command in cmd
    compile_cmd = f'call "{vcvars_path}" && cl.exe /EHsc /O2 /openmp /D_CRT_SECURE_NO_WARNINGS benchmark.cpp /Febenchmark.exe'
    print("\nCompiling benchmark.cpp with MSVC (/O2 /openmp)...")
    t0 = time.time()
    res = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    t1 = time.time()
    
    if res.returncode != 0:
        print("Compilation FAILED!")
        print("Stdout:", res.stdout)
        print("Stderr:", res.stderr)
        return
    print(f"Compilation succeeded in {(t1-t0):.2f} seconds.")
    
    # 3. Run the benchmark executable
    print("\nRunning C++ benchmark executable...")
    run_res = subprocess.run("benchmark.exe", shell=True, capture_output=True, text=True)
    print(run_res.stdout)
    if run_res.stderr:
        print("Stderr:", run_res.stderr)
        
    # 4. Convert output raw binary files back to PNG
    outputs = {
        "hikrobot_cpp_sui.raw": ("hikrobot_cpp_sui.png", "hikrobot_cpp_sui_gamma.png"),
        "hikrobot_cpp_masked.raw": ("hikrobot_cpp_masked.png", "hikrobot_cpp_masked_gamma.png"),
        "hikrobot_cpp_weighted.raw": ("hikrobot_cpp_weighted.png", "hikrobot_cpp_weighted_gamma.png")
    }
    
    H, W = 2048, 2048
    gamma = 2.5
    
    print("\nConverting raw outputs back to PNG images...")
    for raw_file, (png_file, gamma_file) in outputs.items():
        if os.path.exists(raw_file):
            raw_data = np.fromfile(raw_file, dtype=np.float32).reshape(H, W)
            # Clip and save direct output
            img_uint8 = np.clip(raw_data, 0, 255).astype(np.uint8)
            cv2.imwrite(png_file, img_uint8)
            
            # Apply gamma and save gamma version
            img_norm = np.clip(raw_data / 255.0, 0, 1)
            img_gamma = (np.power(img_norm, gamma) * 255.0).astype(np.uint8)
            cv2.imwrite(gamma_file, img_gamma)
            
            print(f"Saved {png_file} and {gamma_file}")
            # Clean up raw file
            os.remove(raw_file)
            
    # Clean up input raw and temp compiler outputs
    if os.path.exists("hikrobot.raw"):
        os.remove("hikrobot.raw")
    for temp_ext in [".obj", ".exe"]:
        temp_file = "benchmark" + temp_ext
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    print("\nBenchmark harness finished successfully!")

if __name__ == "__main__":
    main()
