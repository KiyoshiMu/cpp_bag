from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MK_FEAT_DIR = Path("D:/DATA/mk_feats")


def load_mk_feat(mk_feat_p: Path):
    if not mk_feat_p.exists():
        print(f"{mk_feat_p} does not exist")
        return np.zeros((1, 256), dtype=np.float32)
    with open(mk_feat_p, "r") as f:
        raw = json.load(f)
    arr = np.array([item[2] for item in raw])
    feat = arr.mean(axis=0, keepdims=True)
    return feat
