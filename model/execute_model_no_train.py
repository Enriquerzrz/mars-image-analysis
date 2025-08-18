import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
MODEL_PATH = "models/advanced_cae.h5"
INPUT_FOLDER = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output_png"
OUTPUT_FOLDER = "reconstructed"
LOG_FILE = "anomaly_scores_all.txt"
INPUT_SHAPE = (128, 128, 1)

# === PREPARE OUTPUT ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === LOAD MODEL ===
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH, compile=False)

# === INFERENCE LOOP ===
with open(LOG_FILE, "w") as log:
    for fname in sorted(os.listdir(INPUT_FOLDER)):
        if not fname.endswith(".png"):
            continue

        fpath = os.path.join(INPUT_FOLDER, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Could not read: {fname}")
            continue

        img = cv2.resize(img, INPUT_SHAPE[:2])
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        img_input = np.expand_dims(img, axis=0)  # (1, H, W, 1)

        # Reconstruct and score
        recon = model.predict(img_input, verbose=0)[0]
        score = np.mean((img - recon) ** 2)

        # Save reconstructed image
        recon_uint8 = (recon.squeeze() * 255).astype(np.uint8)
        out_path = os.path.join(OUTPUT_FOLDER, fname)
        cv2.imwrite(out_path, recon_uint8)

        # Write anomaly score
        log.write(f"{fname}\t{score:.6f}\n")

print(f"[DONE] Inference completed.\nReconstructed images: {OUTPUT_FOLDER}\nScores saved to: {LOG_FILE}")
