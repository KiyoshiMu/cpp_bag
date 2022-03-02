from __future__ import annotations

from pathlib import Path

import numpy as np

from cpp_bag import data
from cpp_bag.embed import ds_avg


def test_avg_embed():
    embed_pools = []
    for _ in range(3):
        embed_pools.append(np.random.rand(10, 5))
    embed = np.stack(embed_pools, axis=0)
    assert embed.mean(axis=0).shape == (10, 5)


def test_avg_ds():
    dataset = data.CustomImageDataset(Path("data/feats"), data.LABEL_DIR, 128)
    embed = ds_avg(dataset)
    print(embed.shape)
    assert embed.shape[1] == 256
