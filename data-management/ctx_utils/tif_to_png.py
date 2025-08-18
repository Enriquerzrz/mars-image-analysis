import os
import cv2
import rasterio
import numpy as np
from tqdm import tqdm

# === CONFIGURATION ===
input_base_dir = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output/"
output_base_dir = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output_png/"

# Create output directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

# Get all .tif files in the input directory
tif_files = [f for f in os.listdir(input_base_dir) if f.endswith(".tif")]
print(f"Found {len(tif_files)} TIFF files.")

# Convert each .tif to 8-bit .png
for fname in tqdm(tif_files, desc="Converting TIFF to PNG"):
    fpath = os.path.join(input_base_dir, fname)
    with rasterio.open(fpath) as src:
        img = src.read(1)  # Read first band
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    out_path = os.path.join(output_base_dir, fname.replace(".tif", ".png"))
    cv2.imwrite(out_path, img_uint8)

print("Conversion complete. PNGs saved to:", output_base_dir)
