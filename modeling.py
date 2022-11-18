from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.model_selection import StratifiedShuffleSplit

from cpp_bag import data
from cpp_bag.embed import ds_avg
from cpp_bag.embed import ds_embed
from cpp_bag.embed import ds_project
from cpp_bag.model import BagPooling
from cpp_bag.model import encoder_training
from cpp_bag.plot import measure_slide_vectors

BASE_DIR = "experiments1"

class Planner:
    def __init__(self) -> None:
        self.base = Path(BASE_DIR)
        self.with_mk = True
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        all_cells = data.load_cells()
        self.dataset = data.CustomImageDataset(
            data.FEAT_DIR,
            data.LABEL_DIR,
            bag_size=256,
            cell_threshold=300,
            with_MK=self.with_mk,
            all_cells=all_cells,
        )

    def run(self, n=5):
        sss = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=42)
        x = list(range(len(self.dataset)))
        y = self.dataset.targets
        for trial, (train_index, test_index) in enumerate(sss.split(x, y)):
            marker = str(trial)
            dst_dir = self.base / f"trial{marker}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            split_json_p = dst_dir / f"split{marker}.json"
            with open(split_json_p, "w") as f:
                json.dump(
                    dict(
                        train=train_index.tolist(),
                        val=test_index.tolist(),
                    ),
                    f,
                )

        for trial in range(n):
            dst_dir = self.base / f"trial{trial}"
            split_json_p = dst_dir / f"split{trial}.json"
            model_path = self.train_model(split_json_p, dst_dir)
            make_embeddings(
                model_path,
                split_json_p,
                dataset=self.dataset,
                dst_dir=dst_dir,
                trial=str(trial),
            )

    def train_model(self, split_json_p: str, dst):
        in_dim = 256
        dataset = self.dataset
        with open(split_json_p, "r") as f:
            cache = json.load(f)
            train_indices = cache["train"]
        train_set = data.Subset(dataset, train_indices)
        model_path = encoder_training(
            train_set,
            in_dim=in_dim,
            num_epochs=250,
            num_workers=1,
            dst_dir=dst,
        )
        return model_path


def make_embeddings(
    model_path: str,
    split_json_p: str,
    dst_dir: Path,
    trial: str,
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
        name_mark=f"train{trial}",
    )
    val_pkl_dst, _ = ds_project(
        val_set,
        embed_func,
        dst_dir,
        name_mark=f"val{trial}",
    )
    measure_slide_vectors(
        train_pkl_dst,
        val_pkl_dst,
        mark="pool",
        trial=trial,
        dummy_baseline=True,
        dst=dst_dir,
    )

    avg_func = lambda ds: ds_avg(ds)
    train_avg_pkl_dst, _ = ds_project(
        train_set,
        avg_func,
        dst_dir,
        name_mark=f"train_avg{trial}",
    )
    val_avg_pkl_dst, _ = ds_project(
        val_set,
        avg_func,
        dst_dir,
        name_mark=f"val_avg{trial}",
    )
    measure_slide_vectors(
        train_avg_pkl_dst,
        val_avg_pkl_dst,
        mark="avg",
        trial=trial,
        dst=dst_dir,
        dummy_baseline=False,
    )


if __name__ == "__main__":
    planner = Planner()
    planner.run(n=5)
