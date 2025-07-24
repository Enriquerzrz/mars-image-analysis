import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D,
    Conv2DTranspose, Concatenate, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

class AdvancedCAE:
    def __init__(self, input_shape=(128, 128, 3), latent_dim=64):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.alpha = 0.5
        self._build_model()

    def _build_model(self):
        """Builds a convolutional autoencoder with skip connections."""
        inp = Input(self.input_shape, name="encoder_input")

        # Encoder
        x1 = Conv2D(32, 3, padding='same', activation='relu')(inp)
        x1 = BatchNormalization()(x1)
        p1 = MaxPooling2D()(x1)

        x2 = Conv2D(64, 3, padding='same', activation='relu')(p1)
        x2 = BatchNormalization()(x2)
        p2 = MaxPooling2D()(x2)

        x3 = Conv2D(128, 3, padding='same', activation='relu')(p2)
        x3 = BatchNormalization()(x3)
        p3 = MaxPooling2D()(x3)

        # Bottleneck
        bottleneck = Conv2D(self.latent_dim, 3, padding='same', activation='relu', name="bottleneck")(p3)

        # Decoder with skip connections
        d3 = UpSampling2D()(bottleneck)
        d3 = Conv2DTranspose(128, 3, padding='same', activation='relu')(d3)
        d3 = Concatenate()([d3, x3])

        d2 = UpSampling2D()(d3)
        d2 = Conv2DTranspose(64, 3, padding='same', activation='relu')(d2)
        d2 = Concatenate()([d2, x2])

        d1 = UpSampling2D()(d2)
        d1 = Conv2DTranspose(32, 3, padding='same', activation='relu')(d1)
        d1 = Concatenate()([d1, x1])

        decoded = Conv2DTranspose(self.input_shape[2], 3, padding='same', activation='sigmoid', name="decoder_output")(d1)

        self.model = Model(inp, decoded, name="AdvancedCAE")
        self.encoder = Model(inp, bottleneck, name="encoder_model")

    def mixed_loss(self, y_true, y_pred):
        """Custom loss combining MSE and SSIM."""
        mse = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return mse + self.alpha * (1 - ssim)

    def compile(self, lr=1e-3, alpha=0.5):
        """Compiles the model."""
        self.alpha = alpha
        self.model.compile(optimizer=Adam(lr), loss=self.mixed_loss, metrics=["mse"])

    def _load_images(self, folder):
        """Loads and preprocesses all images in a folder."""
        images = []
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_shape[:2])
            images.append(img.astype(np.float32) / 255.0)
        return np.array(images)

    def train(self, train_dir, val_dir=None, epochs=100, batch_size=32, model_path="adv_cae.h5"):
        """Trains the autoencoder on image datasets."""

        train_images = self._load_images(train_dir)
        if len(train_images) == 0:
            raise ValueError(f"No valid images found in training directory: {train_dir}")

        val_images = None
        if val_dir and os.path.isdir(val_dir) and len(os.listdir(val_dir)) > 0:
            val_images = self._load_images(val_dir)
            if len(val_images) == 0:
                val_images = None

        print(f"Loaded {len(train_images)} training images")
        if val_images is not None:
            print(f"Loaded {len(val_images)} validation images")

        if val_images is not None:
            self.model.fit(train_images, train_images,
                           validation_data=(val_images, val_images),
                           epochs=epochs, batch_size=batch_size)
        else:
            self.model.fit(train_images, train_images,
                           epochs=epochs, batch_size=batch_size)

        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def reconstruct(self, img_array):
        """Reconstructs a single image."""
        return self.model.predict(np.expand_dims(img_array, 0), verbose=0)[0]

    def anomaly_score(self, img):
        """Computes anomaly score based on reconstruction error."""
        rec = self.reconstruct(img)
        return np.mean((img - rec) ** 2)

    def infer_folder(self, folder_in, folder_out=None, log_path="anomaly_log.txt"):
        """Infers and logs anomaly scores from all images in a folder."""
        if folder_out:
            os.makedirs(folder_out, exist_ok=True)

        with open(log_path, 'w') as log:
            for fname in os.listdir(folder_in):
                fpath = os.path.join(folder_in, fname)
                img = cv2.imread(fpath)
                if img is None:
                    print(f"Could not read: {fname}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_norm = cv2.resize(img_rgb, self.input_shape[:2]).astype(np.float32) / 255.0
                score = self.anomaly_score(img_norm)
                log.write(f"{fname}\t{score:.6f}\n")

                if folder_out:
                    rec = (self.reconstruct(img_norm) * 255).astype(np.uint8)
                    rec_bgr = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(folder_out, fname), rec_bgr)

        print(f"Inference complete. Log saved to: {log_path}")
