import json
from pathlib import Path
import zipfile
from tqdm.auto import tqdm

def copy_targets_to_ZipFile(src, dst, img_src:Path):
    with open(src, "r") as f:
        targets = json.load(f)
        
    with zipfile.ZipFile(dst, "w") as z:
        for k, v in  tqdm(targets.items()):
            img_dir = img_src / k
            imgs = [img_dir/f"{i}.jpg" for i in v]
            for img in imgs:
                # keep the same dir structure
                z.write(img, arcname=f"{img_dir.stem}/{img.name}")

if __name__ == "__main__":
    src = Path("cbp_cell_images.json")
    dst = Path("cbp_cell_images.zip")
    img_src = Path("dzi_cellImg/")
    copy_targets_to_ZipFile(src, dst, img_src)