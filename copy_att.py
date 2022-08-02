from pathlib import Path
from PIL import Image
import json


PRED_P = "experiments0/trial2/pool2.json"
DOCS_DIR = Path("D:/DATA/Docs")
DST_DIR = Path("D:/att_rank_copy")
def main(dst_dir = DST_DIR):
    src_dir= Path("att_rank0")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for file in src_dir.glob("*.png"):
        img = Image.open(file)
        new_name = f"{file.stem.split('-')[0]}.jpg"
        dst_file = dst_dir / new_name
        img.save(dst_file, format="JPEG")

def mk_vector_info(pred_p= PRED_P, dst_dir= DST_DIR):
    with open(pred_p, "r") as f:
        pred_data = json.load(f)
    for item in pred_data:
        doc_p = DOCS_DIR / f"{item['index']}.json"
        with open(doc_p, "r") as f:
            data = json.load(f)["result"]
        item["synopsis"] = data
    with open(dst_dir / "pool.json", "w") as f:
        json.dump(pred_data, f)

if __name__ == "__main__":
    # main()
    mk_vector_info()
