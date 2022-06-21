"""Used for prepare the MK feature as Demo's cache
    """

from pathlib import Path


# src_dir = Path("D:\\DATA\\samples")
# files = src_dir.glob("*.smart")
# print([f.stem for f in files])
import shutil

files = ["19_0220_AS", "19_0414_AS", "19_0451_AS", "19_0505_AS", "19_0523_AS"]


def copy_dir(src_dir, dst_dir):
    shutil.copytree(src_dir, dst_dir)


def copy_file(src_file, dst_file):
    shutil.copy(src_file, dst_file)


if __name__ == "__main__":
    # src_dir = Path(
    #     "/storage/create_local/campbell/Campbell_Lab/Research/Asp/data/new_mks/"
    # )
    # dst_dir = Path("/home/muy/wsi/")
    # for f in files:
    #     copy_dir(src_dir / f / "mk_images", dst_dir / "mks" / f )
    src_dir = Path("D:\\DATA\\mk_feats")
    dst_dir = Path("D:\\DATA\\samples\\mks")
    for f in files:
        copy_file(src_dir / f"{f}.json", dst_dir / f"{f}.json")
