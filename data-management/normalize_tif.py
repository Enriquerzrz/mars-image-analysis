import os
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.exposure import match_histograms
from tqdm import tqdm
import cv2

# --- CONFIGURATION ---
tif_path = r".././../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/MurrayLab_CTX_V01_E-004_N-48_Mosaic.tif"
output_folder = r"../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/output_png/"
tile_size = 128
reference_idx = (2, 2)  # Reference tile (row, column) for histogram matching

os.makedirs(output_folder, exist_ok=True)

# --- FUNCTIONS ---

def normalize_min_max(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def normalize_zscore(image):
    return (image - image.mean()) / (image.std() + 1e-8)

# --- MAIN PROCESSING ---

with rasterio.open(tif_path) as src:
    width, height = src.width, src.height
    ncols = (width + tile_size - 1) // tile_size
    nrows = (height + tile_size - 1) // tile_size

    # Load reference tile for histogram matching
    ref_row, ref_col = reference_idx
    ref_window = Window(ref_col * tile_size, ref_row * tile_size, tile_size, tile_size)
    reference_tile = src.read(1, window=ref_window)

    # Process each tile
    for row in tqdm(range(nrows), desc="Rows"):
        for col in range(ncols):
            x = col * tile_size
            y = row * tile_size
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            window = Window(x, y, w, h)
            tile = src.read(1, window=window)

            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile

            matched_tile = match_histograms(tile, reference_tile)
            zscore_tile = normalize_zscore(matched_tile)
            zscore_tile = np.clip(zscore_tile, -3, 3)
            norm_tile = normalize_min_max(zscore_tile)

            # Save as PNG
            filename = f"tile_{row}_{col}.png"
            path = os.path.join(output_folder, filename)
            img_uint8 = (norm_tile * 255).astype(np.uint8)
            cv2.imwrite(path, img_uint8)

print(f"Normalized PNG tiles saved to: {output_folder}")
