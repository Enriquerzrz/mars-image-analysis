import os
import shutil
import random
from tqdm import tqdm

class DatasetSplitter:
    def __init__(self, input_dir, output_base, split_ratios=None, seed=42):
        """
        Initialize the dataset splitter.

        :param input_dir: Path to folder with image tiles.
        :param output_base: Base folder where 'train', 'val', 'test' folders will be created.
        :param split_ratios: Dictionary with split proportions. Default is 70/15/15.
        :param seed: Random seed for reproducibility.
        """
        self.input_dir = input_dir
        self.output_base = output_base
        self.split_ratios = split_ratios or {'train': 0.7, 'val': 0.15, 'test': 0.15}
        self.seed = seed
        self.allowed_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

        os.makedirs(output_base, exist_ok=True)
        for folder in self.split_ratios:
            os.makedirs(os.path.join(output_base, folder), exist_ok=True)

    def _list_images(self):
        return [f for f in os.listdir(self.input_dir)
                if os.path.splitext(f)[1].lower() in self.allowed_exts]

    def split(self):
        """Shuffle and split the dataset into train/val/test folders."""
        images = self._list_images()
        total = len(images)
        print(f"[INFO] Found {total} images.")

        random.seed(self.seed)
        random.shuffle(images)

        n_train = int(total * self.split_ratios['train'])
        n_val = int(total * self.split_ratios['val'])
        n_test = total - n_train - n_val

        split_map = (
            ('train', images[:n_train]),
            ('val', images[n_train:n_train + n_val]),
            ('test', images[n_train + n_val:])
        )

        for split_name, files in split_map:
            split_path = os.path.join(self.output_base, split_name)
            print(f"[INFO] Copying {len(files)} files to {split_name}/")
            for fname in tqdm(files, desc=f"{split_name.upper()}"):
                src = os.path.join(self.input_dir, fname)
                dst = os.path.join(split_path, fname)
                shutil.copy2(src, dst)

        print("[DONE] Dataset successfully split.")

if __name__ == "__main__":
    input_folder = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/output/"
    output_folder = "../../../MurrayLab_GlobalCTXMosaic_V01_E-004_N-32/split_tiles/"

    splitter = DatasetSplitter(input_folder, output_folder)
    splitter.split()
