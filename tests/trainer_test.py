from __future__ import annotations

from pathlib import Path

import torch

from cpp_bag import data
from cpp_bag.model import encoder_training


def test_trainer():
    dataset = data.CustomImageDataset(Path("data/feats"), data.LABEL_DIR, 256)
    size = len(dataset)
    train_size = int(size * 0.8)
    val_size = size - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    encoder_training(train_set, val_set, num_epochs=1)
