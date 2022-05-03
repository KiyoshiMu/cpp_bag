# Partial importance scoring system for slide-level classification:
# 1. K-mean on cell type from samples accross all slides
# 2. Use K-mean to cluster samples into sub-types
# 3. Mask the subtype and make prediction, then the output will be like (subtype, prediction change) pair
# 4. Visualize the subtype and prediction change, Then we can show which subtype is more important for a label
from __future__ import annotations

import pickle
from itertools import chain
from random import sample
from typing import Callable
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans

from cpp_bag import data
from cpp_bag.data import CellInstance
from cpp_bag.data import load_cells
from runner import WITH_MK

K = 3

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


class KMeanMask:
    def __init__(self, k=3) -> None:
        self.kmeans = KMeans(n_clusters=k, random_state=42)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.kmeans, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.kmeans = pickle.load(f)

    def predict(self, X: list[list[float]]) -> list[int]:
        return self.kmeans.predict(X)

    def train(self, X: list[list[float]]) -> None:
        self.kmeans.fit(X)

    @classmethod
    def from_path(cls, path: str, k=3) -> "KMeanMask":
        mask = cls(k=k)
        mask.load(path)
        return mask


class ImScorer:
    def __init__(
        self,
        masks: dict[CellType, KMeanMask],
        predict_func: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.masks = masks
        self.predict_func = predict_func
        all_cells = data.load_cells()
        self.dataset = data.CustomImageDataset(
            data.FEAT_DIR,
            data.LABEL_DIR,
            bag_size=256,
            cell_threshold=300,
            with_MK=WITH_MK,
            all_cells=all_cells,
        )

    def make_masks(self, sample_size=256):
        for cell_type, kmean_mask in self.masks.items():
            print(f"{cell_type}")

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
            kmean_mask.save(f"data/{cell_type}_mask.pkl")

    def predict_mask(self, mask: KMeanMask, cells: list[CellInstance]) -> list[int]:
        X = [cell.feature for cell in cells]
        return mask.predict(X)

    def run() -> dict[tuple[CellType, int], float]:
        pass


def sample_cell_type(cells: list[CellInstance], cell_type: str, sample_size=256):
    """
    Sample cells from a cell type
    """
    type_cells = [cell for cell in cells if cell.label == cell_type]
    if len(type_cells) < sample_size:
        return type_cells
    return sample(type_cells, sample_size)
