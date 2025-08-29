import os
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.exposure import match_histograms
from tqdm import tqdm
import cv2

"""
ES-----------------------------------------------------------------------------------------------

Script: tile_and_save_all_mosaics.py

Descripción:
------------
Este script procesa múltiples mosaicos regionales CTX del dataset del Murray Lab.
Para cada mosaico realiza los siguientes pasos:

1. Localiza el archivo .tif correspondiente dentro de su estructura de carpetas.
2. Divide el mosaico completo en teselas de 128x128 píxeles usando ventanas rasterio.
3. Aplica normalización radiométrica a cada tesela:
   - Ajuste de histograma respecto a una tesela de referencia del mismo mosaico
   - Normalización por puntuación Z para asegurar contraste consistente
   - Normalización min-max para escalar los valores al rango [0, 1]
4. Guarda cada tesela como una imagen .png de 8 bits en un directorio global de salida,
   utilizando nombres de archivo que codifican el origen del mosaico y su posición.

Los nombres de archivo siguen el formato:
    tile_<N-XX>_<fila>_<columna>.png

Ejemplo:
    tile_N-36_012_045.png → tesela de la región N-36, fila 12, columna 45

Este script permite una preprocesamiento unificado entre diferentes zonas orbitales de Marte,
garantizando coherencia visual y estadística de los datos de superficie.
El conjunto resultante está listo para tareas de aprendizaje no supervisado como
autoencoders, detección de anomalías o inspección visual.

Dependencias:
-------------
- numpy
- rasterio
- opencv (cv2)
- scikit-image
- tqdm

EN-----------------------------------------------------------------------------------------------

Script: normalize_tif.py

Description:
------------
This script processes multiple regional CTX mosaics from the Murray Lab dataset.
For each mosaic, it performs the following steps:

1. Locates the corresponding .tif mosaic file inside its nested folder structure.
2. Splits the full mosaic into 128x128 pixel tiles using rasterio windows.
3. Applies radiometric normalization to each tile:
   - Histogram matching against a reference tile from the same mosaic
   - Z-score normalization to ensure consistent contrast
   - Min-max normalization to scale values to [0, 1]
4. Saves each tile as an 8-bit .png image in a single global output directory,
   with filenames that encode their mosaic origin and tile position.

Output filenames follow this format:
    tile_<N-XX>_<row>_<col>.png

Example:
    tile_N-36_012_045.png → tile from region N-36, row 12, column 45

This script enables unified preprocessing across multiple Mars orbital strips,
ensuring consistent visual and statistical representation of surface data.
The resulting dataset is suitable for unsupervised learning tasks such as
autoencoding, anomaly detection, or visual inspection.

Dependencies:
-------------
- numpy
- rasterio
- opencv (cv2)
- scikit-image
- tqdm
"""


# === CONFIGURATION ===
base_dir = "../../../data"
mosaic_names = [
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-32",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-36",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-40",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-44",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-48",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-52",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-64"
]
output_dir = "../../../dataoutput_png_all_mosaics/"
tile_size = 128
reference_tile_idx = (2, 2)

os.makedirs(output_dir, exist_ok=True)

def normalize_min_max(image: np.ndarray) -> np.ndarray:
    """
    Applies min-max normalization to scale pixel values into [0, 1].

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Normalized image.
    """
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def normalize_zscore(image: np.ndarray) -> np.ndarray:
    """
    Applies z-score normalization to ensure consistent contrast across tiles.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Normalized image with zero mean and unit variance.
    """
    return (image - image.mean()) / (image.std() + 1e-8)

def process_mosaic(mosaic_name: str):
    """
    Processes a single CTX mosaic by:
        - Locating the corresponding .tif file
        - Cutting it into 128x128 pixel tiles
        - Normalizing each tile using histogram matching and z-score
        - Saving the result as 8-bit PNG images in a global output folder

    Args:
        mosaic_name (str): Folder name of the mosaic (e.g., "MurrayLab_GlobalCTXMosaic_V01_E-004_N-32").
    """
    tif_path = os.path.join(base_dir, mosaic_name, mosaic_name, f"{mosaic_name.replace('Global', 'CTX')}_Mosaic.tif")
    if not os.path.exists(tif_path):
        print(f"[SKIPPED] TIF not found: {tif_path}")
        return

    print(f"Processing: {mosaic_name}")
    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
        ncols = (width + tile_size - 1) // tile_size
        nrows = (height + tile_size - 1) // tile_size

        # Load reference tile for histogram matching
        ref_row, ref_col = reference_tile_idx
        ref_window = Window(ref_col * tile_size, ref_row * tile_size, tile_size, tile_size)
        try:
            reference_tile = src.read(1, window=ref_window)
            if np.all(reference_tile == 0) or reference_tile.size == 0 or np.std(reference_tile) < 1e-6:
                raise ValueError
        except:
            # Fallback to center tile if reference is invalid
            center_x = src.width // 2
            center_y = src.height // 2
            center_window = Window(center_x, center_y, tile_size, tile_size)
            reference_tile = src.read(1, window=center_window)
            print(f"[WARNING] Fallback reference tile used for {mosaic_name}")

        # Loop over each tile
        for row in tqdm(range(nrows), desc=mosaic_name):
            for col in range(ncols):
                x = col * tile_size
                y = row * tile_size
                w = min(tile_size, width - x)
                h = min(tile_size, height - y)
                window = Window(x, y, w, h)
                tile = src.read(1, window=window)

                # Pad edge tiles
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile

                # Apply normalization pipeline
                matched_tile = match_histograms(tile, reference_tile)
                zscore_tile = normalize_zscore(matched_tile)
                zscore_tile = np.clip(zscore_tile, -3, 3)
                norm_tile = normalize_min_max(zscore_tile)

                # Construct unique filename and save as PNG
                region_id = mosaic_name.split('_')[-1]  # e.g., N-32
                filename = f"tile_{region_id}_{row:03}_{col:03}.png"
                path = os.path.join(output_dir, filename)
                cv2.imwrite(path, (norm_tile * 255).astype(np.uint8))

# === MAIN EXECUTION LOOP ===
for mosaic in mosaic_names:
    process_mosaic(mosaic)

print(f"All mosaics processed. PNG tiles saved to: {output_dir}")
