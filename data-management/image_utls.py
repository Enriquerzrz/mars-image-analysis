import os
import rasterio
import numpy as np
from rasterio.windows import Window
from PIL import Image


def list_geotiff_paths(directory, extension=".tif"):
    """
    Returns a list of all GeoTIFF file paths in the given directory and subdirectories.
    """
    tif_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                tif_paths.append(os.path.join(root, file))
    return tif_paths


def generate_tiles(image_path, tile_size=128, overlap=0):
    """
    Splits a GeoTIFF into smaller tiles.
    Returns a list of (tile_array, row_idx, col_idx) for each tile.
    """
    tiles = []
    with rasterio.open(image_path) as src:
        width = src.width
        height = src.height
        for i in range(0, height - tile_size + 1, tile_size - overlap):
            for j in range(0, width - tile_size + 1, tile_size - overlap):
                window = Window(j, i, tile_size, tile_size)
                tile = src.read(1, window=window)
                if tile.shape == (tile_size, tile_size):
                    tiles.append((tile, i, j))
    return tiles


def save_tiles_as_images(tiles, output_dir, base_name, image_format="png", group_size=10000):
    """
    Saves each tile as a grayscale image into grouped subfolders to avoid file system overload.
    """
    saved = 0
    short_name = base_name.replace("MurrayLab_GlobalCTXMosaic_", "")
    base_output = os.path.join(output_dir, short_name)

    for idx, (tile, row, col) in enumerate(tiles):
        try:
            norm_tile = ((tile - tile.min()) / (tile.max() - tile.min()) * 255).astype(np.uint8)
            image = Image.fromarray(norm_tile)

            # Compute group folder
            group_index = saved // group_size
            group_folder = os.path.join(base_output, f"group_{group_index}")
            os.makedirs(group_folder, exist_ok=True)

            # Save the tile
            filename = f"{short_name}_r{row}_c{col}.{image_format}"
            out_path = os.path.join(group_folder, filename)
            image.save(out_path)

            saved += 1
            if saved % 10 == 0:
                print(f"{saved} tiles saved...")

        except Exception as e:
            print(f"Error saving tile row {row}, col {col}: {e}")

    print(f"Total tiles saved: {saved}")


def process_mosaic_directory(input_dir, output_dir, tile_size=128, overlap=0):
    """
    Processes all .tif files in the input directory and saves tiles to the output directory.
    """
    tif_paths = list_geotiff_paths(input_dir)
    for tif_path in tif_paths:
        base_name = os.path.splitext(os.path.basename(tif_path))[0]
        print(f"Tiling: {base_name}")
        tiles = generate_tiles(tif_path, tile_size, overlap)
        tile_output_dir = os.path.join(output_dir, base_name)
        save_tiles_as_images(tiles, tile_output_dir, base_name)
        print(f"Saved {len(tiles)} tiles from {base_name}.\n")
