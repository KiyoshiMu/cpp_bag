from __future__ import annotations

import json
from pathlib import Path

import torch

from cpp_bag import data
from cpp_bag.embed import ds_avg
from cpp_bag.embed import ds_embed
from cpp_bag.embed import ds_project
from cpp_bag.model import BagPooling
from cpp_bag.model import encoder_training
from cpp_bag.performance import performance_measure


def train_model():
    dataset = data.CustomImageDataset(data.FEAT_DIR, data.LABEL_DIR, 256)
    size = len(dataset)
    print("size:", size)
    train_size = int(size * 0.5)
    val_size = size - train_size
    _train_set, _val_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    with open("data/split.json", "w") as f:
        json.dump(dict(train=_train_set.indices, val=_val_set.indices), f)
    train_set = data.Subset(dataset, _train_set.indices)
    val_set = data.Subset(dataset, _val_set.indices)
    model_path = encoder_training(train_set, val_set, num_epochs=100)
    return model_path


def plotter(model_path: str):
    dataset = data.CustomImageDataset(data.FEAT_DIR, data.LABEL_DIR, 256)
    size = len(dataset)
    print("size:", size)
    with open("data/split.json", "r") as f:
        cache = json.load(f)
        val_indices = cache["val"]
        train_indices = cache["train"]
    val_set = data.Subset(dataset, val_indices)
    train_set = data.Subset(dataset, train_indices)

    model = BagPooling.from_checkpoint(model_path, in_dim=256)
    embed_func = lambda ds: ds_embed(ds, model)
    train_pkl_dst, _ = ds_project(
        train_set,
        embed_func,
        Path("data"),
        name_mark="train",
    )
    val_pkl_dst, _ = ds_project(
        val_set,
        embed_func,
        Path("data"),
        name_mark="val",
    )
    performance_measure(train_pkl_dst, val_pkl_dst, mark="pool", random_base=True)

    avg_func = lambda ds: ds_avg(ds)
    train_avg_pkl_dst, _ = ds_project(
        train_set,
        avg_func,
        Path("data"),
        name_mark="train_avg",
    )
    val_avg_pkl_dst, _ = ds_project(
        val_set,
        avg_func,
        Path("data"),
        name_mark="val_avg",
    )
    performance_measure(train_avg_pkl_dst, val_avg_pkl_dst, mark="avg")


if __name__ == "__main__":
    model_path = train_model()
    plotter(model_path)
    # slide_vectors("data/train_pool.pkl", "data/val_pool.pkl", mark="pool")
