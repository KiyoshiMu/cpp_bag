from pathlib import Path
from PIL import Image
import json

from cpp_bag.label_map import ACCR_LABLE


PRED_P = "experiments0/trial2/pool2.json"
DOCS_DIR = Path("D:/DATA/Docs")
DST_DIR = Path("D:/code/att/public/att_rank")
def main(dst_dir = DST_DIR):
    src_dir= Path("att_rank0")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for file in src_dir.glob("*.png"):
        img = Image.open(file)
        new_name = f"{file.stem.split('-')[0]}.jpg"
        dst_file = dst_dir / new_name
        img.save(dst_file, format="JPEG")

def mk_vector_info(pred_p= PRED_P, dst_dir= DST_DIR, clean=False):
    with open(pred_p, "r") as f:
        pred_data = json.load(f)
    if not clean:
        for item in pred_data:
            doc_p = DOCS_DIR / f"{item['index']}.json"
            with open(doc_p, "r") as f:
                data = json.load(f)["result"]
            item["synopsis"] = data
    if clean:
        for idx, item in enumerate(pred_data):
            att_name = DST_DIR / f"{item['index']}.jpg"
            if att_name.exists():
                att_name.rename(DST_DIR / f"{idx}.jpg")
        pred_data = [{
            "full_label" : item["full_label"],
            "label" :ACCR_LABLE[item["label"]] ,
            "pred" : ACCR_LABLE[item["pred"]],
            "index" : idx,
            "D1" : item["D1"],
            "D2" : item["D2"],
        } for idx, item in enumerate(pred_data) ]
       
    with open(dst_dir / "pool.json", "w") as f:
        json.dump(pred_data, f)

if __name__ == "__main__":
    # main()
    mk_vector_info(clean=True)
