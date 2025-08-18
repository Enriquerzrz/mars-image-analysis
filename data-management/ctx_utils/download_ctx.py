import os
import requests
import urllib3

# Disable SSL certificate verification warnings (we trust this source)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# These CTX tiles have been selected because they cover the geographic region of Mare Acidalium
# Latitudes: ~30°N to 65°N | Longitudes: ~300°E to 360°E (represented here by E-004 ≈ 356°E)
tile_files = [
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-32.zip",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-36.zip",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-40.zip",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-44.zip",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-48.zip",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-52.zip",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-56.zip",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-60.zip",
    "MurrayLab_GlobalCTXMosaic_V01_E-004_N-64.zip"
]

# Base URL for downloading the tiles
base_url = "https://murray-lab.caltech.edu/CTX/V01/tiles"

# Local directory to save the downloaded files
output_dir = r"D:\data\mare_acidalium_ctx"
os.makedirs(output_dir, exist_ok=True)

print("\nStarting CTX tile downloads for Mare Acidalium...\n")

# Loop through each file and download it
for fname in tile_files:
    url = f"{base_url}/{fname}"
    dest_path = os.path.join(output_dir, fname)

    try:
        response = requests.get(url, stream=True, timeout=60, verify=False)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {fname}")
        else:
            print(f"Tile not found on server: {fname}")
    except Exception as e:
        print(f"Error downloading {fname}: {e}")

print("\nDownload process complete.")
