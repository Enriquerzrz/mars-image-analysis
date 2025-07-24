import os
import numpy as np
import cv2

# === CONFIGURATION ===
TILE_SIZE = 128
SCORES_FILE = "../model/anomaly_scores.txt"
RECON_DIR = "../model/reconstructed"
OUTPUT_OVERLAY = "heatmap/heatmap_overlay.png"
OUTPUT_PREVIEW = "heatmap/heatmap_overlay_preview.png"
ALPHA = 0.6  # Opacity for overlay

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
            row = int(name.split("_r")[1].split("_c")[0])
            col = int(name.split("_c")[1])
        except (IndexError, ValueError):
            continue

        rows.append(row)
        cols.append(col)
        tile_scores[(row, col)] = float(score)

# === STEP 2: Determine mosaic size ===
min_row, max_row = min(rows), max(rows)
min_col, max_col = min(cols), max(cols)

height = max_row - min_row + TILE_SIZE
width = max_col - min_col + TILE_SIZE

print(f"Generating overlay image with size: {width} x {height} pixels")

# === STEP 3: Initialize base mosaic and anomaly map ===
base_mosaic = np.zeros((height, width, 3), dtype=np.uint8)
anomaly_layer = np.zeros((height, width), dtype=np.float32)

# === STEP 4: Fill in base mosaic and anomaly scores ===
for (row, col), score in tile_scores.items():
    y = row - min_row
    x = col - min_col
    tile_path = os.path.join(RECON_DIR, f"V01_E-004_N-32_r{row}_c{col}.png")
    tile_img = cv2.imread(tile_path)

    if tile_img is None:
        print(f"Skipping missing tile: {tile_path}")
        continue

    tile_img = cv2.resize(tile_img, (TILE_SIZE, TILE_SIZE))
    base_mosaic[y:y+TILE_SIZE, x:x+TILE_SIZE] = tile_img
    anomaly_layer[y:y+TILE_SIZE, x:x+TILE_SIZE] = score

# === STEP 5: Normalize using percentiles to avoid outliers ===
p5, p95 = np.percentile(list(tile_scores.values()), [5, 95])
norm = np.clip((anomaly_layer - p5) / (p95 - p5 + 1e-8), 0, 1)

# === STEP 6: Create anomaly color layer (red = high) ===
color_map = cv2.applyColorMap((255 - (norm * 255)).astype(np.uint8), cv2.COLORMAP_JET)

# === STEP 7: Overlay color map on base image with transparency ===
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

# === STEP 9: Save reduced version for previewing ===
preview = cv2.resize(overlay_uint8, (overlay.shape[1] // 4, overlay.shape[0] // 4))
cv2.imwrite(OUTPUT_PREVIEW, preview)
print(f"Saved preview image: {OUTPUT_PREVIEW}")
