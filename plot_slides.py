from __future__ import annotations

import pandas as pd

from cpp_bag.plot import slide_vectors


if __name__ == "__main__":
    entropy_df = pd.read_csv("data/resample/resample_preds_stat.csv")
    entropy = entropy_df["entropy"].to_list()
    slide_vectors(
        "data/train_pool.pkl",
        "data/val_pool.pkl",
        val_entropy=entropy,
        mark="slide_vectors",
    )
