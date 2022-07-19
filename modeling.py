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
from cpp_bag.plot import measure_slide_vectors


class Planner:
    def __init__(self) -> None:
        self.base = Path("experiments")
        self.torch_ranGen = torch.Generator().manual_seed(42)
        self.with_mk = True
        print(torch.cuda.is_available())

    def run(self, n=5):
        for trial in range(n):
            marker = str(trial)
            dst_dir = self.base / f"trial{marker}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            split_json_p = dst_dir / f"split{marker}.json"
            model_path, dataset = self.train_model(split_json_p, dst_dir)
            make_embeddings(
                model_path,
                split_json_p,
                dataset=dataset,
                dst_dir=dst_dir,
                marker=marker,
            )

    def train_model(self, split_json_p: str, dst):
        in_dim = 256
        with_MK = self.with_mk
        generator = self.torch_ranGen

        all_cells = data.load_cells()
        dataset = data.CustomImageDataset(
            data.FEAT_DIR,
            data.LABEL_DIR,
            bag_size=256,
            cell_threshold=300,
            with_MK=with_MK,
            all_cells=all_cells,
        )
        size = len(dataset)
        print("size:", size)
        train_size = int(size * 0.5)
        val_size = size - train_size
        _train_set, _val_set = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )
        with open(split_json_p, "w") as f:
            json.dump(dict(train=_train_set.indices, val=_val_set.indices), f)
        train_set = data.Subset(dataset, _train_set.indices)
        model_path = encoder_training(
            train_set,
            in_dim=in_dim,
            num_epochs=250,
            num_workers=1,
            dst_dir=dst,
        )
        return model_path, dataset


def make_embeddings(
    model_path: str,
    split_json_p: str,
    dst_dir: Path,
    marker: str,
    dataset=None,
):
    in_dim = 256
    if dataset is None:
        dataset = data.CustomImageDataset(data.FEAT_DIR, data.LABEL_DIR, 256)
    size = len(dataset)
    print("size:", size)
    with open(split_json_p, "r") as f:
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
        dst_dir,
        name_mark=f"train{marker}",
    )
    val_pkl_dst, _ = ds_project(
        val_set,
        embed_func,
        dst_dir,
        name_mark=f"val{marker}",
    )
    measure_slide_vectors(
        train_pkl_dst,
        val_pkl_dst,
        mark=f"pool{marker}",
        dummy_baseline=True,
        dst=dst_dir,
    )

    avg_func = lambda ds: ds_avg(ds)
    train_avg_pkl_dst, _ = ds_project(
        train_set,
        avg_func,
        dst_dir,
        name_mark=f"train_avg{marker}",
    )
    val_avg_pkl_dst, _ = ds_project(
        val_set,
        avg_func,
        dst_dir,
        name_mark=f"val_avg{marker}",
    )
    measure_slide_vectors(
        train_avg_pkl_dst,
        val_avg_pkl_dst,
        mark=f"avg{marker}",
        dst=dst_dir,
        dummy_baseline=False,
    )

if __name__ == "__main__":
    planner = Planner()
    planner.run(n=5)
