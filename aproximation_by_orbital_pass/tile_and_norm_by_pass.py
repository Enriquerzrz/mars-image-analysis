import os
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.exposure import match_histograms
from tqdm import tqdm
import cv2

# === CONFIGURATION ===
input_dir = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/passes_split_tif/"
output_tile_dir = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/tiles_by_pass/"
tile_size = 128
reference_tile_idx = (2, 2)  # row, col for histogram matching

os.makedirs(output_tile_dir, exist_ok=True)

def normalize_min_max(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def normalize_zscore(image):
    return (image - image.mean()) / (image.std() + 1e-8)

def process_single_tif(tif_path, output_dir):
    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
        ncols = (width + tile_size - 1) // tile_size
        nrows = (height + tile_size - 1) // tile_size
        ref_row, ref_col = reference_tile_idx

        # Try to load reference tile; fallback to center tile if invalid
        try:
            ref_window = Window(ref_col * tile_size, ref_row * tile_size, tile_size, tile_size)
            reference_tile = src.read(1, window=ref_window)

            if reference_tile.size == 0 or np.all(reference_tile == 0) or np.std(reference_tile) < 1e-6:
                raise ValueError("Reference tile is empty, black, or too uniform.")

        except:
            center_x = src.width // 2
            center_y = src.height // 2
            center_window = Window(center_x, center_y, tile_size, tile_size)
            reference_tile = src.read(1, window=center_window)
            print(f"[WARNING] Using fallback reference tile for: {os.path.basename(tif_path)}")

        for row in tqdm(range(nrows), desc=os.path.basename(tif_path)):
            for col in range(ncols):
                x = col * tile_size
                y = row * tile_size
                w = min(tile_size, width - x)
                h = min(tile_size, height - y)
                window = Window(x, y, w, h)
                tile = src.read(1, window=window)

                # Pad if needed
                if tile.shape != (tile_size, tile_size):
                    padded = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded

                # Normalize and match histograms
                matched_tile = match_histograms(tile, reference_tile)
                zscore_tile = normalize_zscore(matched_tile)
                zscore_tile = np.clip(zscore_tile, -3, 3)
                norm_tile = normalize_min_max(zscore_tile)

                # Save as PNG
                out_name = f"tile_{row}_{col}.png"
                pass_folder = os.path.join(output_dir, os.path.splitext(os.path.basename(tif_path))[0])
                os.makedirs(pass_folder, exist_ok=True)
                cv2.imwrite(os.path.join(pass_folder, out_name), (norm_tile * 255).astype(np.uint8))

# === RUN ===
for tif in os.listdir(input_dir):
    if tif.endswith(".tif"):
        tif_path = os.path.join(input_dir, tif)
        process_single_tif(tif_path, output_tile_dir)

print("All passes processed and saved to:", output_tile_dir)
