from __future__ import annotations

import json
import pickle
from collections import defaultdict
from itertools import chain
from math import ceil
from pathlib import Path
from random import sample
from typing import Counter
from typing import NamedTuple
from typing import Sequence

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import thread_map

from cpp_bag.io_utils import simplify_label
from cpp_bag.mk_data import load_mk_feat
from cpp_bag.mk_data import MK_FEAT_DIR

Lineage = set(
    """\
Neutrophil,Metamyelocyte,Myelocyte,\
Promyelocyte,Blast,Erythroblast,Megakaryocyte_nucleus,\
Lymphocyte,Monocyte,Plasma_cell,Eosinophil,Basophil,\
Histiocyte,Mast_cell""".split(
        ",",
    ),
)


class CellInstance(NamedTuple):
    name: str
    label: str
    feature: list[float]


LABEL_DIR = Path("D:/code/Docs")
FEAT_DIR = Path("D:/DATA/feats")


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.labels = dataset.labels[indices]
        self.targets = dataset.targets[indices]
        self.slide_names = dataset.slide_names[indices]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def dump_cells(feat_dir: Path):
    slides = [p for p in feat_dir.glob("*.json")]
    cells = thread_map(_load_cell_feats, slides)
    with open("data/cells.pkl", "wb") as f:
        pickle.dump(cells, f)


def load_cells():
    with open("data/cells.pkl", "rb") as f:
        return pickle.load(f)


def _load_cell_feats(feat_path):
    with open(feat_path, "r") as f:
        feats = [
            cell
            for info in json.load(f)
            if (
                cell := CellInstance(
                    name=info[0],
                    label=info[1],
                    feature=info[2][:256],
                )
            ).label
            in Lineage
        ]
        return feats


class CustomImageDataset(Dataset):
    def __init__(
        self,
        feat_dir: Path,
        label_dir: Path,
        bag_size=256,
        cell_threshold=300,
        with_MK=True,
        all_cells=None,
    ):
        _slides = [p for p in feat_dir.glob("*.json")]
        if all_cells is None:
            all_cells = thread_map(self._load_feats, _slides)
        _p_cells = [
            (p, cells)
            for p, cells in zip(_slides, all_cells)
            if len(cells) > cell_threshold
        ]
        _slide_names = np.array([p.stem for p, _ in _p_cells])
        _labels = np.array(
            [self._load_doc(label_dir / f"{name}.json") for name in _slide_names],
        )
        _simple_labels = np.array([simplify_label(l) for l in _labels])
        self.slide_names = _slide_names
        self.labels = _labels
        self.le = LabelEncoder()
        self.targets = self.le.fit_transform(_simple_labels)
        self.slide_portion: list[dict[str, int]] = [
            self._mk_portion([cell.label for cell in cells], bag_size)
            for _, cells in _p_cells
        ]
        print("\nslide_portion:", self.slide_portion[:3])
        self.features = [
            torch.as_tensor([cell.feature for cell in cells], dtype=torch.float32)
            for _, cells in _p_cells
        ]
        self.cell_groups = [
            self._group_by_label([cell.label for cell in cells])
            for _, cells in _p_cells
        ]
        self.with_MK = with_MK
        if with_MK:
            self.MK_features = [
                torch.as_tensor(
                    load_mk_feat(MK_FEAT_DIR / f"{slide_name}.json"),
                    dtype=torch.float32,
                )
                for slide_name in _slide_names
            ]

    def _load_feats(self, feat_path):
        with open(feat_path, "r") as f:
            feats = [
                cell
                for info in json.load(f)
                if (
                    cell := CellInstance(
                        name=info[0],
                        label=info[1],
                        feature=info[2][:256],
                    )
                ).label
                in Lineage
            ]
            return feats

    def _load_doc(self, doc_path):
        with open(doc_path, "r") as f:
            return json.load(f)["tags"]

    def _mk_portion(self, labels: list[str], size: int):
        assert len(labels) >= size, "Not enough cells"
        counter = Counter(labels)
        ratio = len(labels) / size
        out = {
            k: targe if ((targe := ceil(v / ratio)) > 1) else 1
            for k, v in counter.items()
        }
        more = sum(out.values()) - size
        if more > 0:
            ranks = sorted(
                [item for item in out.items() if item[1] > 1],
                key=lambda x: x[1],
                reverse=True,
            )
            for i in range(more):
                out[ranks[i % len(ranks)][0]] -= 1
        assert sum(out.values()) == size, f"Not match: {sum(out.values())} != {size}"
        return out

    def _group_by_label(self, labels: list[str]):

        group = defaultdict(list)
        for idx, label in enumerate(labels):
            group[label].append(idx)
        return group

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        label = self.targets[idx]
        slide_portion = self.slide_portion[idx]
        group = self.cell_groups[idx]
        feature_bag = torch.index_select(
            self.features[idx],
            0,
            torch.as_tensor(self._sample_idx(slide_portion, group)),
        )
        if self.with_MK:
            feature = torch.cat([feature_bag, self.MK_features[idx]], dim=0)
        else:
            feature = feature_bag
        return feature, label

    def _sample_idx(self, slide_portion: dict[str, int], group: dict[str, list[int]]):

        out = list(
            chain.from_iterable(
                sample(group[k], k=v) for k, v in slide_portion.items()
            ),
        )
        return out
