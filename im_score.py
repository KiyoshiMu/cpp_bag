# Partial importance scoring system for slide-level classification:
# 1. K-mean on cell type from samples accross all slides
# 2. Use K-mean to cluster samples into sub-types
# 3. Mask the subtype and make prediction, then the output will be like (subtype, prediction change) pair
# 4. Visualize the subtype and prediction change, Then we can show which subtype is more important for a label
from __future__ import annotations

import json
import pickle
from itertools import chain
from pathlib import Path
from random import sample
from typing import Callable
from typing import Literal
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from cpp_bag import data
from cpp_bag.data import CellInstance
from cpp_bag.embed import ds_embed
from cpp_bag.embed import ds_project
from cpp_bag.io_utils import pkl_load
from cpp_bag.io_utils import simplify_label
from cpp_bag.model import BagPooling
from cpp_bag.performance import create_knn
from cpp_bag.plot import heat_map
from runner import WITH_MK

SUBTYPE_K = 2

CellType = Literal[
    "Neutrophil",
    "Metamyelocyte",
    "Myelocyte",
    "Promyelocyte",
    "Blast",
    "Erythroblast",
    "Megakaryocyte_nucleus",
    "Lymphocyte",
    "Monocyte",
    "Plasma_cell",
    "Eosinophil",
    "Basophil",
    "Histiocyte",
    "Mast_cell",
]

CELL_TYPES: list[CellType] = [
    "Neutrophil",
    "Metamyelocyte",
    "Myelocyte",
    "Promyelocyte",
    "Blast",
    "Erythroblast",
    "Megakaryocyte_nucleus",
    "Lymphocyte",
    "Monocyte",
    "Plasma_cell",
    "Eosinophil",
    "Basophil",
    "Histiocyte",
    "Mast_cell",
]


class KMeanMask:
    def __init__(self, k=SUBTYPE_K) -> None:
        self.kmeans = KMeans(n_clusters=k, random_state=42)

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.kmeans, f)

    def load(self, path) -> None:
        with open(path, "rb") as f:
            self.kmeans = pickle.load(f)

    def predict(self, X: list[list[float]]) -> np.ndarray:
        return self.kmeans.predict(X)

    def train(self, X: list[list[float]]) -> None:
        self.kmeans.fit(X)


class ImScorer:
    def __init__(
        self,
        embed_func: Callable[[Dataset], np.ndarray],
        masks: Optional[dict[CellType, KMeanMask]],
    ) -> None:
        self.embed_func = embed_func
        all_cells = data.load_cells()
        self.dataset = data.CustomImageDataset(
            data.FEAT_DIR,
            data.LABEL_DIR,
            bag_size=256,
            cell_threshold=300,
            with_MK=WITH_MK,
            all_cells=all_cells,
        )
        self.masks = (
            masks
            if masks is not None
            else {k: KMeanMask(k=SUBTYPE_K) for k in CELL_TYPES}
        )

    def make_masks(
        self,
        dst: Path,
        sample_size=256,
        use_cache=True,
    ):
        for cell_type, kmean_mask in self.masks.items():
            print(f"{cell_type}")
            kmean_mask_p = dst / f"{cell_type}_kmean_mask.pkl"
            if use_cache and kmean_mask_p.exists():
                kmean_mask.load(kmean_mask_p)
                continue

            cells = list(
                chain(
                    *(
                        sample_cell_type(cell, cell_type, sample_size=sample_size)
                        for cell in self.dataset.cells
                    ),
                ),
            )
            X = [cell.feature for cell in cells]
            kmean_mask.train(X)
            kmean_mask.save(kmean_mask_p)

    def _predict_mask(self, mask: KMeanMask, cells: list[CellInstance]) -> np.ndarray:
        X = [cell.feature for cell in cells]
        return mask.predict(X)

    def run(self, cell_type: CellType, subtype: int, split_json_p: str):
        slide_names = self.dataset.slide_names
        mask_map = {}
        mask_dst = Path("mask/result")
        mask_dst.mkdir(exist_ok=True, parents=True)
        for slide_idx, name in enumerate(slide_names):
            slide_cells = self.dataset.cells[slide_idx]
            target_cells = [
                (idx, cell)
                for idx, cell in enumerate(slide_cells)
                if cell.label == cell_type
            ]
            pred_subtypes = self._predict_mask(
                self.masks[cell_type],
                [cell[1] for cell in target_cells],
            )
            mask_flag = np.zeros(len(slide_cells), dtype=bool)
            mask_idx = [
                idx
                for ((idx, _), pred_subtype) in zip(target_cells, pred_subtypes)
                if pred_subtype == subtype
            ]
            mask_flag[mask_idx] = True
            mask_map[name] = torch.as_tensor(mask_flag, dtype=torch.bool)
        self.dataset.mask_map = mask_map

        with open(split_json_p, "r") as f:
            cache = json.load(f)
            val_indices = cache["val"]
            # train_indices = cache["train"]
        val_set = data.Subset(self.dataset, val_indices)
        val_pkl_dst, _ = ds_project(
            val_set,
            self.embed_func,
            mask_dst,
            name_mark=f"val_mask_{cell_type}_{subtype}",
        )
        return val_pkl_dst


class ImAnalyst:
    def __init__(self, train_pkl_p, val_pkl_p) -> None:
        train = pkl_load(train_pkl_p)
        val = pkl_load(val_pkl_p)
        labels = [simplify_label(l) for l in train["labels"]]
        refer_embed = train["embed_pool"]
        self.knn = create_knn(refer_embed, labels)
        self.classes_ = self.knn.classes_
        val_embed = val["embed_pool"]
        self.pred_probs: np.ndarray = self.knn.predict_proba(val_embed)
        self.pred_labels: np.ndarray = self.knn.predict(val_embed)

    def diff(self, ret_pkl_p):
        ret = pkl_load(ret_pkl_p)
        ret_embed = ret["embed_pool"]
        ret_pred_probs: np.ndarray = self.knn.predict_proba(ret_embed)
        diff = ret_pred_probs - self.pred_probs
        diff_df = pd.DataFrame(diff, columns=self.classes_)
        diff_df["label"] = self.pred_labels
        return diff_df

    def summary(self, diff_dir: Path):
        """For each label, how the mask will influence the prediction"""
        diff_df_csvs = list(diff_dir.glob("*_mask_*.csv"))
        mask_effect = {}
        for csv in diff_df_csvs:
            diff_df = pd.read_csv(csv)
            diff_mean = diff_df.groupby("label").mean()
            effects = []
            for label in self.classes_:
                effects.append(diff_mean.loc[label, label])
            mask_name = csv.stem.removeprefix("val_").removesuffix("_pool")
            mask_effect[mask_name] = list(effects)
        mask_effect_df = pd.DataFrame(mask_effect, index=self.classes_).transpose()
        print(mask_effect_df)
        return mask_effect_df


def make_mask_embed(model_path: str, split_json_p="data/0/split0.json"):
    in_dim = 256
    model = BagPooling.from_checkpoint(model_path, in_dim=in_dim)
    embed_func = lambda ds: ds_embed(ds, model)
    scorer = ImScorer(embed_func, None)
    mask_dir = Path("mask")
    mask_dir.mkdir(exist_ok=True, parents=True)
    scorer.make_masks(mask_dir)
    for cell_type in CELL_TYPES:
        for subtype in range(SUBTYPE_K):
            print(f"{cell_type}_{subtype}")
            try:
                dst = scorer.run(cell_type, subtype, split_json_p)
                print(dst)
            except Exception as e:
                print(e)


def diff_mask_embed(ret_dir_p: str, dst=Path("mask/mask_diff")):
    ret_dir = Path(ret_dir_p)
    ret_pkls = ret_dir.glob("*.pkl")
    im_analyst = ImAnalyst("data/0/train0_pool.pkl", "data/0/val0_pool.pkl")
    dst.mkdir(exist_ok=True, parents=True)
    for ret_pkl in ret_pkls:
        print(ret_pkl)
        diff_df = im_analyst.diff(ret_pkl)
        fn = ret_pkl.stem
        diff_df.to_csv(dst / f"{fn}.csv")
    mask_effect_df = im_analyst.summary(dst)
    mask_effect_df.to_csv(dst / "mask_effect.csv")


def sample_cell_type(cells: list[CellInstance], cell_type: str, sample_size=256):
    """
    Sample cells from a cell type
    """
    type_cells = [cell for cell in cells if cell.label == cell_type]
    if len(type_cells) < sample_size:
        return type_cells
    return sample(type_cells, sample_size)


def mask_effect_heat_map(effect_csv_p):
    mask_effect_df = pd.read_csv(effect_csv_p, index_col=0)
    values = mask_effect_df.values * -1
    # annotation text is the :.3f string of the values
    annotation_text = values.round(3).astype(str)
    fig = heat_map(
        z=values,
        x=mask_effect_df.columns.to_list(),
        y=mask_effect_df.index.to_list(),
        annotation_text=annotation_text,
    )
    fig.write_image("mask_effect.png", scale=2)


if __name__ == "__main__":
    make_mask_embed("data/0/pool-1650909062154.pth")
    diff_mask_embed("mask/result")
    mask_effect_heat_map("mask/mask_diff/mask_effect.csv")
