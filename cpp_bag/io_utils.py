from __future__ import annotations

import json
import pickle
from pathlib import Path
from zipfile import ZipFile


def simplify_label(label: str):
    if "normal" in label:
        out = "normal"
    elif "leukemia" in label and "acute" in label:
        out = "acute leukemia"
    elif "myelodysplastic syndrome" in label:
        out = "myelodysplastic syndrome"
    elif "plasma" in label:
        out = "plasma"
    elif "lymphoproliferative disorder" in label:
        out = "lymphoproliferative disorder"
    else:
        out = "other"
    return out


def pkl_load(src_p):
    with open(src_p, "rb") as target:
        out = pickle.load(target)
    return out


def pkl_dump(obj, out_p):
    with open(out_p, "wb") as target:
        pickle.dump(obj, target)


def json_dump(obj, dst_p):
    with open(dst_p, "w", encoding="utf8") as target:
        json.dump(obj, target)


def json_load(src_p):
    with open(src_p, "r", encoding="utf8") as target:
        docs = json.load(target)
    return docs


def collect_files(dir_p: str, ext=".json"):
    _dir_p = Path(dir_p)
    return _dir_p.rglob(f"*{ext}")


def mk_out_path(dst, name, mkdir=True):
    dst_p = Path(dst)
    if mkdir:
        dst_p.mkdir(exist_ok=True)
    return dst_p / name


def unzip_file(src, dst):
    Path(dst).mkdir(exist_ok=False)
    with ZipFile(src, "r") as in_f:
        in_f.extractall(dst)
