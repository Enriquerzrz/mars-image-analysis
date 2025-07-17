import os
import sys
from image_utls import generate_tiles, save_tiles_as_images

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python tile_single_mosaic.py <mosaic_base_name>")
        print("Example: python tile_single_mosaic.py MurrayLab_GlobalCTXMosaic_V01_E-004_N-32")
        sys.exit(1)

    # Mosaic file name provided by user
    base_name = sys.argv[1]

    # Configuration
    input_dir = r"D:\data\mare_acidalium_ctx\extracted"
    output_dir = r"D:\data\mare_acidalium_ctx\tiles"
    tile_size = 128
    overlap = 0

    # Full path to .tif file
    tif_path = os.path.join(input_dir, base_name, base_name, f"MurrayLab_CTX_V01_E-004_N-32_Mosaic.tif")

    if not os.path.exists(tif_path):
        print(f"GeoTIFF not found: {tif_path}")
        sys.exit(1)

    print(f"Tiling: {base_name}")
    tiles = generate_tiles(tif_path, tile_size, overlap)

    tile_output_dir = os.path.join(output_dir, base_name)
    save_tiles_as_images(tiles, tile_output_dir, base_name)

    print(f"Finished processing {base_name}.")

if __name__ == "__main__":
    main()
