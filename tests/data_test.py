from __future__ import annotations

from pathlib import Path
from typing import Counter

import numpy as np
import pytest
import torch

from cpp_bag import data


@pytest.mark.skip
def test_data_init():
    dataset = data.CustomImageDataset(Path("data/feats"), data.LABEL_DIR, 256)
    feature_bag, label = dataset[0]
    assert feature_bag.size() == (257, 256)
    print(label)


@pytest.mark.skip
def test_data_differ():
    dataset = data.CustomImageDataset(Path("data/feats"), data.LABEL_DIR, 256)
    feature_bag1, _ = dataset[0]
    feature_bag2, _ = dataset[0]
    assert [v for v in feature_bag1.numpy()[0]] != [v for v in feature_bag2.numpy()[0]]


def test_data_mask():
    dataset = data.CustomImageDataset(Path("data/feats"), data.LABEL_DIR, 256)
    cell_type = "Blast"
    dataset.setup_mask(cell_type)
    feature, _, sample_cell_labels = dataset.example_samples(0)
    count = 0
    for idx, cell_label in enumerate(sample_cell_labels):
        if cell_label == cell_type:
            masked_cell_feature = feature[idx]
            feature_sum = masked_cell_feature.sum()
            print(masked_cell_feature)
            assert feature_sum < 128
            count += 1
    assert count > 100
