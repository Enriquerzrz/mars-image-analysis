# Re-run the conversion using a safer output directory with write permissions
import os
import cv2
import rasterio
import numpy as np
from tqdm import tqdm

# Use a temporary writable directory
output_base_dir = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output_png/"
input_base_dir = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output/"

# Define dataset splits
splits = ["train", "val", "test"]

# Create output directories
for split in splits:
    os.makedirs(os.path.join(output_base_dir, split), exist_ok=True)

# Convert float32 TIFF tiles to 8-bit PNG format
for split in splits:
    input_dir = os.path.join(input_base_dir, split)
    output_dir = os.path.join(output_base_dir, split)

    tif_files = [f for f in os.listdir(input_dir) if f.endswith(".tif")]
    print(f"Processing {split} ({len(tif_files)} files)...")

    for fname in tqdm(tif_files, desc=f"Converting {split}"):
        fpath = os.path.join(input_dir, fname)
        with rasterio.open(fpath) as src:
            img = src.read(1)  # Read first band
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(output_dir, fname.replace(".tif", ".png"))
        cv2.imwrite(out_path, img_uint8)

print("Conversion complete. PNGs saved to:", output_base_dir)
