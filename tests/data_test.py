from __future__ import annotations

from pathlib import Path

from cpp_bag import data


def test_data_init():
    dataset = data.CustomImageDataset(Path("data/feats"), data.LABEL_DIR, 256)
    feature_bag, label = dataset[0]
    assert feature_bag.size() == (256, 256)
    print(label)


def test_data_differ():
    dataset = data.CustomImageDataset(Path("data/feats"), data.LABEL_DIR, 256)
    feature_bag1, _ = dataset[0]
    feature_bag2, _ = dataset[0]
    assert [v for v in feature_bag1.numpy()[0]] != [v for v in feature_bag2.numpy()[0]]
