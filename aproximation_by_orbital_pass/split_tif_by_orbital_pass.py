import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# === CONFIGURATION ===
tif_path = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/MurrayLab_CTX_V01_E-004_N-48_Mosaic.tif"
shapefile_path = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/MurrayLab_CTX_V01_E-004_N-48_SeamMap.shp"
output_dir = "../../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-48/passes_split_tif/"
os.makedirs(output_dir, exist_ok=True)

# === LOAD SHAPEFILE ===
print("Loading seam map...")
seam_gdf = gpd.read_file(shapefile_path)
print(f"Found {len(seam_gdf)} orbital footprints.")

# === OPEN TIF ===
with rasterio.open(tif_path) as src:
    for idx, row in seam_gdf.iterrows():
        geom = [row["geometry"]]

        try:
            out_image, out_transform = mask(src, geom, crop=True)
        except Exception as e:
            print(f"[ERROR] Skipping polygon {idx}: {e}")
            continue

        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        out_path = os.path.join(output_dir, f"passada_{idx+1:02}.tif")
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"Saved: {out_path}")

print("All orbital passes exported.")
