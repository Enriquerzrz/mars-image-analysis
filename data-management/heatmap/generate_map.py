import os
import numpy as np
import cv2

# === CONFIGURATION ===
TILE_SIZE = 128
SCORES_FILE = "../../model/anomaly_scores_26082025.txt"
RECON_DIR = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/output_png_all/"
OUTPUT_OVERLAY = "heatmap_overlay.png"
OUTPUT_PREVIEW = "heatmap_overlay_preview.png"
ALPHA = 0.6  # Overlay transparency

# === STEP 1: Read anomaly scores and positions ===
tile_scores = {}
rows, cols = [], []

with open(SCORES_FILE, "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        fname, score = parts
        if not fname.endswith(".png"):
            continue
        name = os.path.splitext(fname)[0]
        try:
            _, row, col = name.split("_")  # tile_0_15
            row = int(row)
            col = int(col)
        except (IndexError, ValueError):
            continue

        rows.append(row)
        cols.append(col)
        tile_scores[(row, col)] = float(score)

# === STEP 2: Determine mosaic size ===
min_row, max_row = min(rows), max(rows)
min_col, max_col = min(cols), max(cols)

height = (max_row - min_row + 1) * TILE_SIZE
width = (max_col - min_col + 1) * TILE_SIZE

print(f"Generating overlay image with size: {width} x {height} pixels")

# === STEP 3: Initialize base mosaic and anomaly map ===
base_mosaic = np.zeros((height, width, 3), dtype=np.uint8)
anomaly_layer = np.zeros((height, width), dtype=np.float32)

# === STEP 4: Fill in base mosaic and anomaly scores ===
loaded_tiles = 0
missing_tiles = 0

for (row, col), score in tile_scores.items():
    y = (row - min_row) * TILE_SIZE
    x = (col - min_col) * TILE_SIZE
    tile_path = os.path.join(RECON_DIR, f"tile_{row}_{col}.png")
    tile_img = cv2.imread(tile_path)

    if tile_img is None:
        print(f"Skipping missing tile: {tile_path}")
        missing_tiles += 1
        continue

    tile_img = cv2.resize(tile_img, (TILE_SIZE, TILE_SIZE))
    base_mosaic[y:y+TILE_SIZE, x:x+TILE_SIZE] = tile_img
    anomaly_layer[y:y+TILE_SIZE, x:x+TILE_SIZE] = np.float32(score)
    loaded_tiles += 1

print(f"\nTiles successfully loaded: {loaded_tiles}")
print(f"Tiles missing: {missing_tiles} of {len(tile_scores)} total\n")

# === STEP 5: Normalize anomaly scores using percentiles (to reduce outlier effect) ===
p5, p95 = np.percentile(np.array(list(tile_scores.values()), dtype=np.float32), [5, 95])
p5 = np.float32(p5)
p95 = np.float32(p95)
print(f"Anomaly score percentiles: P5 = {p5:.6f}, P95 = {p95:.6f}")

norm = np.clip((anomaly_layer - p5) / (p95 - p5 + np.float32(1e-8)), 0, 1).astype(np.float32)

# === STEP 6: Apply color map (JET) → red = high anomaly ===
color_map = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

# === STEP 7: Overlay color map on base image ===
overlay = cv2.addWeighted(color_map, ALPHA, base_mosaic, 1 - ALPHA, 0)

# === STEP 8: Save full-resolution image ===
try:
    assert overlay.ndim == 3 and overlay.shape[2] == 3, "Overlay must be RGB"
    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
    success = cv2.imwrite(OUTPUT_OVERLAY, overlay_uint8)
    if not success:
        raise IOError("cv2.imwrite failed")
    print(f"Saved full overlay image: {OUTPUT_OVERLAY}")
except Exception as e:
    print(f"Error saving overlay image: {e}")

# === STEP 9: Save a preview (1/4 resolution) ===
preview = cv2.resize(overlay_uint8, (overlay.shape[1] // 4, overlay.shape[0] // 4))
cv2.imwrite(OUTPUT_PREVIEW, preview)
print(f"Saved preview image: {OUTPUT_PREVIEW}")
