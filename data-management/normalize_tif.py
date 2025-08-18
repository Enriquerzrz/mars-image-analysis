import os
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.exposure import match_histograms
from tqdm import tqdm

# --- CONFIGURATION ---
tif_path = ".././../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/MurrayLab_CTX_V01_E-004_N-32_Mosaic.tif"        # Input TIF image
output_folder = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output/"         # Output folder for processed tiles
tile_size = 128                            # Tile size in pixels (square tiles)
reference_idx = (2, 2)                     # Reference tile coordinates (row, column)

os.makedirs(output_folder, exist_ok=True)

# --- FUNCTIONS ---

def normalize_min_max(image):
    """Apply Min-Max normalization to scale image values to [0, 1] range."""
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def save_tile(tile, meta, filename):
    """Save a single tile to disk as a GeoTIFF file."""
    meta.update({
        "height": tile.shape[0],
        "width": tile.shape[1],
        "count": 1,
        "dtype": 'float32'
    })
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(tile, 1)

# --- MAIN PROCESSING ---

with rasterio.open(tif_path) as src:
    width, height = src.width, src.height
    ncols = (width + tile_size - 1) // tile_size
    nrows = (height + tile_size - 1) // tile_size
    meta = src.meta.copy()

    # Load reference tile for histogram matching
    ref_row, ref_col = reference_idx
    ref_window = Window(ref_col * tile_size, ref_row * tile_size, tile_size, tile_size)
    reference_tile = src.read(1, window=ref_window)

    # Loop over each tile in the image
    for row in tqdm(range(nrows), desc="Rows"):
        for col in range(ncols):
            x = col * tile_size
            y = row * tile_size
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            window = Window(x, y, w, h)
            tile = src.read(1, window=window)

            # Pad if tile is smaller than tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile

            # Step 1: Match histogram to reference tile
            matched_tile = match_histograms(tile, reference_tile)

            # Step 2: Normalize tile to [0, 1]
            normalized_tile = normalize_min_max(matched_tile)

            # Step 3: Save normalized tile
            filename = f"tile_{row}_{col}.tif"
            path = os.path.join(output_folder, filename)
            save_tile(normalized_tile.astype(np.float32), meta, path)

print(f"Normalized tiles saved to: {output_folder}")