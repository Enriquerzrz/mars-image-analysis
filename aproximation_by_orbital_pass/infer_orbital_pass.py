import os
import sys
sys.path.append("../model")

from advanced_cae import AdvancedCAE

# === CONFIGURATION ===
model_path = "../model/models/advanced_cae_26082025.h5"
input_base = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/tiles_by_pass/"
output_reco_base = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/reconstructed_by_pass/"
output_scores_base = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/scores_by_pass/"
input_shape = (128, 128, 1)

os.makedirs(output_reco_base, exist_ok=True)
os.makedirs(output_scores_base, exist_ok=True)

# === LOAD MODEL ===
cae = AdvancedCAE(input_shape=input_shape, latent_dim=128)
cae.model.load_weights(model_path)

# === INFER ON EACH PASS ===
for folder in sorted(os.listdir(input_base)):
    folder_path = os.path.join(input_base, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing: {folder}")

    out_reco = os.path.join(output_reco_base, folder)
    out_scores = os.path.join(output_scores_base, f"anomaly_scores_{folder}.txt")
    os.makedirs(out_reco, exist_ok=True)

    cae.infer_folder(folder_path, folder_out=out_reco, log_path=out_scores)

print("All passes processed.")
