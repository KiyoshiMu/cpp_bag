# Partial importance scoring system for slide-level classification:
# 1. K-mean on cell type from samples accross all slides
# 2. Use K-mean to cluster samples into sub-types
# 3. Mask the subtype and make prediction, then the output will be like (subtype, prediction change) pair
# 4. Visualize the subtype and prediction change, Then we can show which subtype is more important for a label
from __future__ import annotations

import json
import pickle
from itertools import chain
from math import ceil
from pathlib import Path
from random import sample
from typing import Callable
from typing import Literal
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset

from cpp_bag import data
from cpp_bag.data import CellInstance
from cpp_bag.embed import ds_embed
from cpp_bag.embed import ds_project
from cpp_bag.io_utils import json_dump
from cpp_bag.io_utils import json_load
from cpp_bag.io_utils import pkl_load
from cpp_bag.io_utils import simplify_label
from cpp_bag.model import BagPooling
from cpp_bag.plot import heat_map

WITH_MK = True
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
    "Megakaryocyte",
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
    "Megakaryocyte",
]


class KMeanMask:
    def __init__(self, k=3) -> None:
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
        subtype_k: int = 3,
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
            enable_mask=True,
        )
        self.masks = (
            masks
            if masks is not None
            else {k: KMeanMask(k=subtype_k) for k in CELL_TYPES}
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

    def run(self, cell_type: CellType, subtype: int, split_json_p: str, dst: Path):
        slide_names = self.dataset.slide_names
        mask_map = {}
        mask_dst = dst / "result"
        mask_dst.mkdir(exist_ok=True, parents=True)
        for slide_idx, name in enumerate(slide_names):
            slide_cells = self.dataset.cells[slide_idx]
            target_cells = [
                (idx, cell)
                for idx, cell in enumerate(slide_cells)
                if cell.label == cell_type
            ]
            mask_flag = np.zeros(len(slide_cells), dtype=bool)
            if len(target_cells) > 0:
                pred_subtypes = self._predict_mask(
                    self.masks[cell_type],
                    [cell[1] for cell in target_cells],
                )
                mask_idx = [
                    idx
                    for ((idx, _), pred_subtype) in zip(target_cells, pred_subtypes)
                    if pred_subtype == subtype
                ]
                mask_flag[mask_idx] = True
            mask_map[name] = torch.as_tensor(mask_flag, dtype=torch.bool)
        self.dataset._mask_map = mask_map

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

    def run_on_type(self, cell_type: CellType, split_json_p: str, dst: Path):
        mask_dst = dst / "result"
        mask_dst.mkdir(exist_ok=True, parents=True)
        mask_info = self.dataset.setup_mask(cell_type)
        json_dump(mask_info, mask_dst / f"{cell_type}_mask_info.json")
        with open(split_json_p, "r") as f:
            cache = json.load(f)
            val_indices = cache["val"]
            # train_indices = cache["train"]
        val_set = data.Subset(self.dataset, val_indices)
        val_pkl_dst, _ = ds_project(
            val_set,
            self.embed_func,
            mask_dst,
            name_mark=f"val_mask_{cell_type}",
        )
        return val_pkl_dst


class Centroid:
    def __init__(self, refer_embed, labels):
        self.clf = NearestCentroid()
        self.clf.fit(refer_embed, labels)
        self.centroids_ = self.clf.centroids_
        self.classes_ = self.clf.classes_

    def predict_proba(self, embed):
        return pairwise_distances(embed, self.centroids_, metric="cosine")


class ImAnalyst:
    def __init__(self, train_pkl_p, val_pkl_p) -> None:
        train = pkl_load(train_pkl_p)
        val = pkl_load(val_pkl_p)
        labels = [simplify_label(l) for l in train["labels"]]
        refer_embed = train["embed_pool"]
        # self.measurement = create_knn(refer_embed, labels)
        self.measurement = Centroid(refer_embed, labels)
        self.classes_ = self.measurement.classes_
        val_embed = val["embed_pool"]
        self.val_labels = [simplify_label(l) for l in val["labels"]]
        self.pred_probs: np.ndarray = self.measurement.predict_proba(val_embed)
        # self.pred_labels: np.ndarray = self.measurement.predict(val_embed)
        self.slide_names = val["index"]

    def diff(self, ret_pkl_p):
        ret = pkl_load(ret_pkl_p)
        ret_embed = ret["embed_pool"]
        ret_pred_probs: np.ndarray = self.measurement.predict_proba(ret_embed)
        diff = ret_pred_probs - self.pred_probs
        diff_df = pd.DataFrame(diff, columns=self.classes_)
        # diff_df["label"] = self.pred_labels
        diff_df["label"] = self.val_labels
        diff_df.index = self.slide_names
        return diff_df

    def summary(self, diff_dir: Path):
        """For each label, how the mask will influence the prediction"""

        def _get_cell_type(csv_p):
            c_type = csv_p.stem.split("_mask_")[-1].removesuffix("_pool")
            return c_type

        diff_df_csvs = list(diff_dir.glob("*_mask_*.csv"))
        cell_types = [_get_cell_type(csv_p) for csv_p in diff_df_csvs]
        slide_names = pd.read_csv(diff_df_csvs[0], index_col=0).index.to_list()
        # !TODO ugly code
        mask_cell_count = load_mask_cell_count(cell_types, slide_names)
        print(mask_cell_count)
        min_mask_cell_count = round(len(slide_names) * 0.5)
        print(min_mask_cell_count)
        mask_cell_count = {
            cell_type: count
            for cell_type, count in mask_cell_count.items()
            if count > min_mask_cell_count
        }
        mask_effect = {}
        for csv_p in diff_df_csvs:
            diff_df = pd.read_csv(csv_p, index_col=0)
            cell_type = _get_cell_type(csv_p)
            if cell_type not in mask_cell_count:
                continue

            effects = diff_df.iloc[:, :6].sum(axis=0) / mask_cell_count[cell_type]
            # print(effects)
            mask_name = _get_cell_type(csv_p)
            mask_effect[mask_name] = list(effects)
        mask_effect_df = pd.DataFrame(mask_effect, index=self.classes_).transpose()
        print(mask_effect_df)
        return mask_effect_df


def load_mask_cell_count(cell_types, slide_names, dst=Path("mask_n0/result/")):
    """Load the mask cell count for each type"""
    mask_cell_count = {
        cell_type: sum(
            _load_mask_info(dst / f"{cell_type}_mask_info.json", slide_names).values(),
        )
        for cell_type in cell_types
    }
    return mask_cell_count


def _load_mask_info(p: Path, slide_names, bag_size=256):
    ### output: {slide_name: mask_cell_count}
    def _get_sample_size(info):
        if info["total_count"] == 0 or info["mask_count"] == 0:
            return 0
        if info["total_count"] == 1:  # Megakaryocyte
            return 2
        ratio = info["mask_count"] / info["total_count"]
        # data.py line 182
        size = ceil(ratio * bag_size)
        if size < 1:
            return 1

        assert size < bag_size, f"{p}|  {size} < {bag_size}"
        return size

    data = json_load(p)
    infos = {
        slide_name: _get_sample_size(data[slide_name]) for slide_name in slide_names
    }
    return infos


def make_mask_embed(
    model_path: str,
    split_json_p: str,
    mask_dir=Path("mask"),
    subtype_k=0,
):
    in_dim = 256
    model = BagPooling.from_checkpoint(model_path, in_dim=in_dim)
    embed_func = lambda ds: ds_embed(ds, model)
    scorer = ImScorer(embed_func, None)
    mask_dir.mkdir(exist_ok=True, parents=True)
    if subtype_k == 0:
        for cell_type in CELL_TYPES:
            dst = scorer.run_on_type(cell_type, split_json_p, mask_dir)
            print(dst)
    else:
        scorer.make_masks(mask_dir)
        for cell_type in CELL_TYPES:
            for subtype in range(subtype_k):
                print(f"{cell_type}_{subtype}")
                dst = scorer.run(cell_type, subtype, split_json_p, mask_dir)
                print(dst)


def diff_mask_embed(ret_dir_p: str, train_pkl_p, val_pkl_p, dst=Path("mask/mask_diff")):
    ret_dir = Path(ret_dir_p)
    ret_pkls = ret_dir.glob("*.pkl")
    im_analyst = ImAnalyst(train_pkl_p, val_pkl_p)
    dst.mkdir(exist_ok=True, parents=True)
    for ret_pkl in ret_pkls:
        # print(ret_pkl)
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


def mask_effect_heat_map(effect_csv_p, dst, use_scale=True):
    mask_effect_df = pd.read_csv(effect_csv_p, index_col=0)
    mask_effect_df.drop(["other"], axis=1, inplace=True)
    # values = mask_effect_df.values * -1
    # annotation text is the :.3f string of the values
    # annotation_text = values.round(3).astype(str)
    labels = mask_effect_df.columns.to_list()
    mask_names: list[str] = [
        name.removeprefix("mask_") for name in mask_effect_df.index
    ]
    rank_values = []
    rank = [idx for idx in range(len(mask_names))]
    annotation_text = []
    for label in labels:
        mask_effect = mask_effect_df.loc[:, label]
        if use_scale:
            mask_effect = minmax_scale(mask_effect)
        sorted_idx = np.argsort(mask_effect)
        rank_values.append([mask_effect[idx] for idx in sorted_idx])
        annotation_text.append([mask_names[idx] for idx in sorted_idx])
    fig = heat_map(
        z=np.transpose(rank_values),
        x=[ACCR_LABLE[l] for l in labels],
        y=rank,
        annotation_text=np.transpose(annotation_text),
    )
    fig.write_image(dst)


ACCR_LABLE = {
    "normal": "NORMAL",
    "acute leukemia": "ACL",
    "lymphoproliferative disorder": "LPD",
    "myelodysplastic syndrome": "MDS",
    "plasma": "PCN",
}


def mask_effect_cell_ordered(effect_csv_p, dst, use_scale=True):
    mask_effect_df = pd.read_csv(effect_csv_p, index_col=0)
    mask_effect_df.drop(["other"], axis=1, inplace=True)
    labels = [ACCR_LABLE[l] for l in mask_effect_df.columns]
    mask_names: list[str] = [
        name.removeprefix("mask_") for name in mask_effect_df.index
    ]
    rank_values = []
    rank = [idx for idx in range(len(labels))]
    annotation_text = []
    for cell_type in mask_effect_df.index:
        mask_effect = mask_effect_df.loc[cell_type, :]
        if use_scale:
            mask_effect = minmax_scale(mask_effect)
        sorted_idx = np.argsort(mask_effect)
        rank_values.append([mask_effect[idx] for idx in sorted_idx])
        annotation_text.append([labels[idx] for idx in sorted_idx])
    fig = heat_map(
        z=np.transpose(rank_values),
        x=mask_names,
        y=rank,
        annotation_text=np.transpose(annotation_text),
    )
    fig.write_image(dst)


if __name__ == "__main__":
    # make_mask_embed(
    #     "experiments0/trial2/pool-1659028139226.pth",
    #     "experiments0/trial2/split2.json",
    #     mask_dir=Path("mask_n0"),
    #     subtype_k=0,
    # )
    # diff_mask_embed("mask_n0/result", "experiments0/trial2/train2_pool.pkl", "experiments0/trial2/val2_pool.pkl", dst=Path("mask_n0/mask_diff"))
    # mask_effect_heat_map("mask_n0/mask_diff/mask_effect.csv", "mask_n0/mask_effect.pdf")
    mask_effect_cell_ordered(
        "mask_n0/mask_diff/mask_effect.csv", "mask_n0/mask_effect_cell.pdf",
    )
    # pd.read_csv("mask_n0/mask_diff/mask_effect.csv").rename(ACCR_LABLE, axis=1).to_latex(
    #     "mask_n0/mask_effect.tex", index=False, float_format="%.3e"
    # )
