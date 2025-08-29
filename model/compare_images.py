import os
import cv2
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim

# === CONFIGURATION ===
original_dir = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output_png/"
reconstructed_dir = "reconstructed/"
scores_file = "anomaly_scores_22082025.txt"
output_dir = "comparison_visuals/"
use_ssim = True
top_k = 30  # Número de candidatos para seleccionar aleatoriamente

os.makedirs(output_dir, exist_ok=True)

# === LOAD SCORES ===
tile_scores = {}
with open(scores_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        fname, score = parts
        tile_scores[fname] = float(score)

# === SELECT RANDOM NORMAL AND ANOMALOUS TILES ===
sorted_tiles = sorted(tile_scores.items(), key=lambda x: x[1])
normals = sorted_tiles[:top_k]
anomals = sorted_tiles[-top_k:]

best_tile = random.choice(normals)[0]
worst_tile = random.choice(anomals)[0]
selected_tiles = [("normal", best_tile), ("anomalous", worst_tile)]

# === PROCESS SELECTED TILES ===
for label, fname in selected_tiles:
    path_orig = os.path.join(original_dir, fname)
    path_reco = os.path.join(reconstructed_dir, fname)

    # Load grayscale images
    orig = cv2.imread(path_orig, cv2.IMREAD_GRAYSCALE)
    reco = cv2.imread(path_reco, cv2.IMREAD_GRAYSCALE)
    if orig is None or reco is None:
        print(f"[WARNING] Skipping {fname} (missing file)")
        continue

    orig_norm = orig.astype(np.float32) / 255.0
    reco_norm = reco.astype(np.float32) / 255.0

    # MSE per pixel
    error = (orig_norm - reco_norm) ** 2

    if use_ssim:
        ssim_map, _ = ssim(orig_norm, reco_norm, full=True, data_range=1.0)
        error += (1 - ssim_map)

    error_vis = (error / error.max() * 255).astype(np.uint8)
    error_color = cv2.applyColorMap(255 - error_vis, cv2.COLORMAP_JET)

    # Stack original, reconstructed, and error map
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    reco_rgb = cv2.cvtColor(reco, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack((orig_rgb, reco_rgb, error_color))

    # Add labels
    cv2.putText(comparison, "Original", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(comparison, "Reconstructed", (130, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(comparison, "Error Map", (300, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    out_path = os.path.join(output_dir, f"comparison_{label}.png")
    cv2.imwrite(out_path, comparison)

print(f"Visual comparisons saved in: {output_dir}")
