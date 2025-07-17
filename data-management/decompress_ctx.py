import os
import zipfile

# Directory where the ZIP files are stored
zip_dir = r"D:\data\mare_acidalium_ctx"

# Directory where the contents will be extracted
extract_dir = os.path.join(zip_dir, "extracted")
os.makedirs(extract_dir, exist_ok=True)

print("\nStarting extraction of CTX mosaic files...\n")

# Loop through each ZIP file and extract
for filename in os.listdir(zip_dir):
    if filename.endswith(".zip"):
        zip_path = os.path.join(zip_dir, filename)
        tile_name = filename.replace(".zip", "")
        output_path = os.path.join(extract_dir, tile_name)

        os.makedirs(output_path, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            print(f"Extracted: {filename} → {tile_name}/")
        except zipfile.BadZipFile:
            print(f"Invalid ZIP file: {filename}")
        except Exception as e:
            print(f"Error extracting {filename}: {e}")

print("\nAll files processed.")
