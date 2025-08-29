import os
import cv2
import numpy as np

# === CONFIGURATION ===
TILE_SIZE = 128
RECON_DIR = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output_png/"
SCORES_FILE = "../../model/anomaly_scores_all.txt"
OUTPUT_IMAGE = "plain_overlay.png"
OUTPUT_PREVIEW = "plain_overlay_preview.png"

# === STEP 1: Leer coordenadas desde los nombres ===
tile_positions = {}
rows, cols = [], []

with open(SCORES_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        fname, _ = parts
        if not fname.endswith(".png"):
            continue
        name = os.path.splitext(fname)[0]
        try:
            _, row, col = name.split("_")
            row = int(row)
            col = int(col)
        except Exception:
            continue

        rows.append(row)
        cols.append(col)
        tile_positions[(row, col)] = fname

# === STEP 2: Crear mosaico vacío ===
min_row, max_row = min(rows), max(rows)
min_col, max_col = min(cols), max(cols)

n_rows = max_row - min_row + 1
n_cols = max_col - min_col + 1
height = n_rows * TILE_SIZE
width = n_cols * TILE_SIZE

mosaic = np.zeros((height, width, 3), dtype=np.uint8)

# === STEP 3: Insertar las tiles ===
tiles_loaded = 0
for (row, col), fname in tile_positions.items():
    y = (row - min_row) * TILE_SIZE
    x = (col - min_col) * TILE_SIZE
    tile_path = os.path.join(RECON_DIR, fname)

    tile_img = cv2.imread(tile_path)
    if tile_img is None:
        print(f"[WARNING] Missing tile: {fname}")
        continue

    tile_img = cv2.resize(tile_img, (TILE_SIZE, TILE_SIZE))
    mosaic[y:y+TILE_SIZE, x:x+TILE_SIZE] = tile_img
    tiles_loaded += 1

print(f"Tiles successfully inserted: {tiles_loaded}")

# === STEP 4: Guardar resultado completo y preview ===
cv2.imwrite(OUTPUT_IMAGE, mosaic)
print(f"Saved plain overlay: {OUTPUT_IMAGE}")

preview = cv2.resize(mosaic, (width // 4, height // 4))
cv2.imwrite(OUTPUT_PREVIEW, preview)
print(f"Saved preview: {OUTPUT_PREVIEW}")
