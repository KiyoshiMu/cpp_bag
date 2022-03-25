from __future__ import annotations

import numpy as np

from cpp_bag.mk_data import load_mk_feat
from cpp_bag.mk_data import MK_FEAT_DIR


def test_mk_data():
    feat = load_mk_feat(MK_FEAT_DIR / "18_0002_AS.json")
    assert feat.shape == (1, 256)
    bag_arr = np.random.rand(256, 256)
    new_arr = np.concatenate([bag_arr, feat], axis=0)
    assert new_arr.shape == (257, 256)
