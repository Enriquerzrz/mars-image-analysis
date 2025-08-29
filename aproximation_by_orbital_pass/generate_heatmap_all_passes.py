import os
import numpy as np
import cv2

# === CONFIGURATION ===
score_dir = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/scores_by_pass/"
reco_dir_base = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/reconstructed_by_pass/"
output_dir = "heatmaps/"
tile_size = 128
alpha = 0.6

os.makedirs(output_dir, exist_ok=True)

# === FIND ALL PASADA FILES ===
score_files = [f for f in os.listdir(score_dir) if f.startswith("anomaly_scores_passada_") and f.endswith(".txt")]

for score_file in sorted(score_files):
    passada_name = score_file.replace("anomaly_scores_", "").replace(".txt", "")
    scores_path = os.path.join(score_dir, score_file)
    recon_dir = os.path.join(reco_dir_base, passada_name)

    print(f"🔍 Generating heatmap for {passada_name}...")

    tile_scores = {}
    rows, cols = [], []

    with open(scores_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            fname, score = parts
            name = os.path.splitext(fname)[0]
            try:
                _, row, col = name.split("_")
                row = int(row)
                col = int(col)
            except:
                continue
            tile_scores[(row, col)] = float(score)
            rows.append(row)
            cols.append(col)

    if not tile_scores:
        print(f"[WARNING] No valid scores found in {score_file}")
        continue

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    height = (max_row - min_row + 1) * tile_size
    width = (max_col - min_col + 1) * tile_size

    base_mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    anomaly_layer = np.zeros((height, width), dtype=np.float32)

    for (row, col), score in tile_scores.items():
        y = (row - min_row) * tile_size
        x = (col - min_col) * tile_size
        tile_path = os.path.join(recon_dir, f"tile_{row}_{col}.png")
        tile_img = cv2.imread(tile_path)

        if tile_img is None:
            print(f"[WARNING] Missing tile: {tile_path}")
            continue

        tile_img = cv2.resize(tile_img, (tile_size, tile_size))
        base_mosaic[y:y+tile_size, x:x+tile_size] = tile_img
        anomaly_layer[y:y+tile_size, x:x+tile_size] = np.float32(score)

    # Normalize using percentiles
    p5, p95 = np.percentile(list(tile_scores.values()), [5, 95])
    norm = np.clip((anomaly_layer - p5) / (p95 - p5 + 1e-8), 0, 1).astype(np.float32)

    # Apply color map and overlay
    color_map = cv2.applyColorMap((255 - (norm * 255)).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(color_map, alpha, base_mosaic, 1 - alpha, 0)

    # Save
    overlay_path = os.path.join(output_dir, f"heatmap_{passada_name}.png")
    preview_path = os.path.join(output_dir, f"heatmap_{passada_name}_preview.png")
    cv2.imwrite(overlay_path, overlay)
    preview = cv2.resize(overlay, (overlay.shape[1] // 4, overlay.shape[0] // 4))
    cv2.imwrite(preview_path, preview)

    print(f"Saved: {overlay_path}")

print("All heatmaps generated.")
