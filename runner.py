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
from cpp_bag.plot import slide_vectors

MARK = "0"
DST = Path("data")
SPLIT_JSON = DST / f"split{MARK}.json"
WITH_MK = True
TORCH_RANDOM_G = torch.Generator().manual_seed(42)


def init_global(trial: int):
    global MARK
    global DST
    global SPLIT_JSON
    MARK = str(trial)
    DST = Path("data") / MARK
    DST.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON = DST / f"split{MARK}.json"


def train_model():
    in_dim = 256
    all_cells = data.load_cells()
    dataset = data.CustomImageDataset(
        data.FEAT_DIR,
        data.LABEL_DIR,
        bag_size=256,
        cell_threshold=300,
        with_MK=WITH_MK,
        all_cells=all_cells,
    )
    size = len(dataset)
    print("size:", size)
    train_size = int(size * 0.5)
    val_size = size - train_size
    _train_set, _val_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=TORCH_RANDOM_G,
    )
    with open(SPLIT_JSON, "w") as f:
        json.dump(dict(train=_train_set.indices, val=_val_set.indices), f)
    train_set = data.Subset(dataset, _train_set.indices)
    val_set = data.Subset(dataset, _val_set.indices)
    model_path = encoder_training(
        train_set,
        val_set,
        in_dim=in_dim,
        num_epochs=250,
        num_workers=1,
        dst_dir=DST,
    )
    return model_path, dataset


def plotter(model_path: str, dataset=None):
    in_dim = 256
    if dataset is None:
        dataset = data.CustomImageDataset(data.FEAT_DIR, data.LABEL_DIR, 256)
    size = len(dataset)
    print("size:", size)
    with open(SPLIT_JSON, "r") as f:
        cache = json.load(f)
        val_indices = cache["val"]
        train_indices = cache["train"]
    val_set = data.Subset(dataset, val_indices)
    train_set = data.Subset(dataset, train_indices)

    model = BagPooling.from_checkpoint(model_path, in_dim=in_dim)
    embed_func = lambda ds: ds_embed(ds, model)
    train_pkl_dst, _ = ds_project(
        train_set,
        embed_func,
        DST,
        name_mark=f"train{MARK}",
    )
    val_pkl_dst, _ = ds_project(
        val_set,
        embed_func,
        DST,
        name_mark=f"val{MARK}",
    )
    performance_measure(
        train_pkl_dst,
        val_pkl_dst,
        mark=f"pool{MARK}",
        random_base=True,
    )

    avg_func = lambda ds: ds_avg(ds)
    train_avg_pkl_dst, _ = ds_project(
        train_set,
        avg_func,
        DST,
        name_mark=f"train_avg{MARK}",
    )
    val_avg_pkl_dst, _ = ds_project(
        val_set,
        avg_func,
        DST,
        name_mark=f"val_avg{MARK}",
    )
    performance_measure(train_avg_pkl_dst, val_avg_pkl_dst, mark=f"avg{MARK}")


if __name__ == "__main__":
    for i in range(5):
        # if i == 0:
        #     torch.rand(10, generator=TORCH_RANDOM_G)
        #     continue
        init_global(i)
        # model_path, dataset = train_model()
        # plotter(model_path, dataset)
        slide_vectors(
            DST / f"train{MARK}_pool.pkl",
            DST / f"val{MARK}_pool.pkl",
            mark=MARK,
            dst=DST,
        )
