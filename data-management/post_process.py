import os
import cv2
import numpy as np
import rasterio
import random
from skimage.exposure import match_histograms
from tqdm import tqdm

# ----------- Image Processing Functions -----------

def is_black_tile(tile, threshold=0.98):
    """Check if a tile is mostly black based on intensity threshold."""
    black_pixels = np.sum(tile == 0)
    total_pixels = tile.size
    ratio = black_pixels / total_pixels
    if ratio > threshold:
        print(f"[SKIPPED - BLACK] Ratio: {ratio:.2f}")
    return ratio > threshold

def normalize_min_max(tile):
    """Scale pixel values to [0, 1] range using min-max normalization."""
    return (tile - tile.min()) / (tile.max() - tile.min() + 1e-8)

def resize_tile(tile, target_size=(128, 128)):
    """Resize tile to the target input size using interpolation."""
    return cv2.resize(tile, target_size, interpolation=cv2.INTER_AREA)

def preprocess_tile(tile, reference_tile=None, target_size=(128, 128)):
    """Full preprocessing for a tile: histogram match, normalize, resize."""
    if is_black_tile(tile):
        return None
    if reference_tile is not None:
        tile = match_histograms(tile, reference_tile)
    tile = normalize_min_max(tile)
    tile = resize_tile(tile, target_size)
    return tile

# ----------- Utility -----------

def get_split_subfolder(split_ratios):
    """Return 'train', 'val', or 'test' based on split ratios."""
    r = random.random()
    if r < split_ratios['train']:
        return 'train'
    elif r < split_ratios['train'] + split_ratios['val']:
        return 'val'
    else:
        return 'test'

def load_reference_tile(reference_tile_path, fallback_paths):
    """Load reference tile or fall back to first valid tile from list."""
    if reference_tile_path and os.path.exists(reference_tile_path):
        try:
            with rasterio.open(reference_tile_path) as ref:
                tile = ref.read(1)
                if not is_black_tile(tile):
                    print(f"[INFO] Loaded reference tile: {reference_tile_path}")
                    return tile
                else:
                    print(f"[WARNING] Reference tile is black, skipping it.")
        except Exception as e:
            print(f"[ERROR] Could not load reference tile: {e}")

    # Fallback: try to find the first usable tile
    for path in fallback_paths:
        try:
            with rasterio.open(path) as ref:
                tile = ref.read(1)
                if not is_black_tile(tile):
                    print(f"[INFO] Using fallback reference tile: {path}")
                    return tile
        except Exception:
            continue
    print("[WARNING] No valid reference tile found.")
    return None

# ----------- Main Processing Function -----------

def process_tiles(input_dir, output_dir_base, reference_tile_path=None, target_size=(128, 128)):
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    for folder in split_ratios:
        os.makedirs(os.path.join(output_dir_base, folder), exist_ok=True)

    log_path = os.path.join(output_dir_base, "processing_log.txt")
    total, saved, skipped = 0, 0, 0
    log_lines = []

    filenames = sorted([f for f in os.listdir(input_dir) if f.endswith(".tif")])
    full_paths = [os.path.join(input_dir, f) for f in filenames]
    print(f"Found {len(filenames)} TIF tiles.")

    # Load reference or fallback tile
    reference_tile = load_reference_tile(reference_tile_path, full_paths)

    for fname in tqdm(filenames, desc="Processing Tiles"):
        fpath = os.path.join(input_dir, fname)
        try:
            with rasterio.open(fpath) as src:
                tile = src.read(1)
        except Exception as e:
            log_lines.append(f"[ERROR] Failed to read {fname}: {e}")
            continue

        processed = preprocess_tile(tile, reference_tile, target_size)
        if processed is not None:
            split = get_split_subfolder(split_ratios)
            out_path = os.path.join(output_dir_base, split, fname.replace(".tif", ".png"))
            cv2.imwrite(out_path, (processed * 255).astype(np.uint8))
            saved += 1
            log_lines.append(f"[SAVED-{split.upper()}] {fname}")
        else:
            skipped += 1
            log_lines.append(f"[SKIPPED] {fname}")
        total += 1

    print(f"Finished. Total: {total}, Saved: {saved}, Skipped: {skipped}")
    with open(log_path, "w") as log_file:
        log_file.write(f"Total: {total}, Saved: {saved}, Skipped: {skipped}\n\n")
        log_file.write("\n".join(log_lines))

# ----------- Entry Point -----------

if __name__ == "__main__":
    input_dir = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output/"
    output_dir_base = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output/dataset_splits_128"

    reference_tile_path = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/MurrayLab_CTX_V01_E-004_N-32_Mosaic.tif"

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    process_tiles(input_dir, output_dir_base, reference_tile_path)
