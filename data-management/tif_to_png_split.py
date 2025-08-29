import os
import cv2
import rasterio
import numpy as np
import random
from tqdm import tqdm

# === CONFIGURATION ===
input_base_dir = r"../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/output_tif/"
output_base_dir = r"../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/output_png_split/"
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
use_fraction = 0.5  # Use only 50% of the tiles

# === CREATE OUTPUT FOLDERS ===
for split in split_ratios:
    os.makedirs(os.path.join(output_base_dir, split), exist_ok=True)

# === COLLECT AND SHUFFLE FILES ===
tif_files = [f for f in os.listdir(input_base_dir) if f.endswith(".tif")]
random.shuffle(tif_files)

subset_size = int(len(tif_files) * use_fraction)
tif_files = tif_files[:subset_size]

print(f"Processing {len(tif_files)} tiles with split ratios {split_ratios}...")

# === PROCESS EACH TILE ===
for fname in tqdm(tif_files, desc="Converting and splitting tiles"):
    fpath = os.path.join(input_base_dir, fname)
    try:
        with rasterio.open(fpath) as src:
            img = src.read(1)  # Read first band
    except Exception as e:
        print(f"[ERROR] Skipping {fname}: {e}")
        continue

    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

    # Assign split
    r = random.random()
    if r < split_ratios['train']:
        split = 'train'
    elif r < split_ratios['train'] + split_ratios['val']:
        split = 'val'
    else:
        split = 'test'

    out_path = os.path.join(output_base_dir, split, fname.replace(".tif", ".png"))
    cv2.imwrite(out_path, img_uint8)

print("Conversion and dataset split complete.")
print("Output saved to:", output_base_dir)
