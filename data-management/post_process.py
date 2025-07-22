import os
import cv2
import numpy as np

# ----------- Image Processing Functions -----------

# Checks if a tile is mostly black based on a pixel intensity threshold
def is_black_tile(tile, threshold=0.98):
    black_pixels = np.sum(tile == 0)
    total_pixels = tile.size
    return black_pixels / total_pixels > threshold

# Normalizes pixel values to the range [0, 1]
def normalize_tile(tile):
    return tile / 255.0

# Applies histogram equalization to improve image contrast
def equalize_histogram(tile):
    if len(tile.shape) == 3 and tile.shape[2] == 3:
        tile_yuv = cv2.cvtColor(tile, cv2.COLOR_BGR2YUV)
        tile_yuv[:, :, 0] = cv2.equalizeHist(tile_yuv[:, :, 0])
        return cv2.cvtColor(tile_yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(tile)

# Resizes the image to the target dimensions for model input
def resize_tile(tile, target_size=(128, 128)):
    return cv2.resize(tile, target_size, interpolation=cv2.INTER_AREA)

# Complete preprocessing pipeline for a single tile
def preprocess_tile(tile, target_size=(128, 128)):
    if is_black_tile(tile):
        return None
    tile = equalize_histogram(tile)
    tile = normalize_tile(tile)
    tile = resize_tile(tile, target_size)
    return tile

# ----------- Group Processing Function -----------

def process_group(group_name, input_dir, output_dir, target_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "processing_log.txt")

    total = 0
    saved = 0
    skipped = 0
    log_lines = []

    filenames = sorted(os.listdir(input_dir))
    num_files = len(filenames)
    print(f"Processing {group_name}: {num_files} tiles")

    for i, fname in enumerate(filenames, start=1):
        fpath = os.path.join(input_dir, fname)
        tile = cv2.imread(fpath)

        if tile is None:
            log_lines.append(f"[ERROR] Could not read: {fname}")
            continue

        processed = preprocess_tile(tile, target_size)
        if processed is not None:
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, (processed * 255).astype(np.uint8))
            saved += 1
            log_lines.append(f"[SAVED] {fname}")
        else:
            skipped += 1
            log_lines.append(f"[SKIPPED - BLACK] {fname}")

        total += 1
        if i % 50 == 0 or i == num_files:
            print(f"  [{i}/{num_files}] Processed: {total}, Saved: {saved}, Skipped: {skipped}")

    print(f"Finished {group_name}. Total: {total}, Saved: {saved}, Skipped: {skipped}\n")

    with open(log_path, "w") as log_file:
        log_file.write(f"Group: {group_name}\nTotal: {total}, Saved: {saved}, Skipped: {skipped}\n\n")
        log_file.write("\n".join(log_lines))

# ----------- Main Execution -----------

if __name__ == "__main__":
    input_base_dir = r"D:\data\mare_acidalium_ctx\tiles\MurrayLab_GlobalCTXMosaic_V01_E-004_N-32\V01_E-004_N-32"
    output_base_dir = r"D:\data\mare_acidalium_ctx\preprocessed\V01_E-004_N-32"

    # List all "group_*" folders
    if not os.path.exists(input_base_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_base_dir}")

    group_dirs = [d for d in os.listdir(input_base_dir) if d.startswith("group_")]
    if not group_dirs:
        print("No group_* directories found.")
    else:
        for group in sorted(group_dirs):
            input_dir = os.path.join(input_base_dir, group)
            output_dir = os.path.join(output_base_dir, group)
            process_group(group, input_dir, output_dir)
