from __future__ import annotations

from pathlib import Path

from cpp_bag import data


def test_data_init():
    dataset = data.CustomImageDataset(Path("data/feats"), data.LABEL_DIR, 256)
    feature_bag, label = dataset[0]
    feature_bag.size == (256, 256)
    print(label)
