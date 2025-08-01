import os
from advanced_cae import AdvancedCAE

def main():
    # -------- Configuration --------
    input_shape = (128, 128, 3)
    latent_dim = 128
    learning_rate = 1e-4
    ssim_alpha = 0.7
    epochs = 50

    # Dataset paths
    train_dir = "../dataset/train"
    val_dir = "../dataset/validacion"
    test_dir = "../dataset/test"

    # Output paths
    model_path = "models/advanced_cae.h5"
    recon_output_dir = "reconstructed"
    log_file = "anomaly_scores.txt"

    # -------- Preparations --------
    os.makedirs("models", exist_ok=True)
    os.makedirs(recon_output_dir, exist_ok=True)

    for path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset directory not found: {path}")

    # -------- Model Initialization --------
    print("Initializing AdvancedCAE model...")
    cae = AdvancedCAE(input_shape=input_shape, latent_dim=latent_dim)
    cae.compile(lr=learning_rate, alpha=ssim_alpha)

    # -------- Training --------
    print("Starting training...")
    cae.train(train_dir=train_dir, val_dir=val_dir, epochs=epochs, model_path=model_path)

    # -------- Inference --------
    print("Running inference on test set...")
    cae.infer_folder(test_dir, folder_out=recon_output_dir, log_path=log_file)

    print("All done")

if __name__ == "__main__":
    main()
