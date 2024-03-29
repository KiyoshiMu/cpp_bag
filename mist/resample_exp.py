from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier

from cpp_bag import data
from cpp_bag.embed import ds_embed
from cpp_bag.embed import ds_project
from cpp_bag.io_utils import pkl_load
from cpp_bag.io_utils import simplify_label
from cpp_bag.model import BagPooling
from cpp_bag.performance import dump_metric


def resampler(resample_times=10):
    with_MK = True
    in_dim = 256
    all_cells = data.load_cells()
    dataset = data.CustomImageDataset(
        data.FEAT_DIR,
        data.LABEL_DIR,
        bag_size=256,
        cell_threshold=256,
        with_MK=with_MK,
        all_cells=all_cells,
    )
    size = len(dataset)
    print("size:", size)
    with open("data/split.json", "r") as f:
        cache = json.load(f)
        val_indices = cache["val"]
    val_set = data.Subset(dataset, val_indices)

    mp = "pool-1648142022566_MK.pth"
    model = BagPooling.from_checkpoint(mp, in_dim=in_dim)
    embed_func = lambda ds: ds_embed(ds, model)
    for i in range(resample_times):
        ds_project(
            val_set,
            embed_func,
            name_mark=f"val-{i}",
            resample_times=1,
            dst_dir=Path("data/resample"),
        )


def embed_diff(resample_times=10):

    train_pkl_p = Path("data/train_pool.pkl")
    train = pkl_load(train_pkl_p)
    labels = [simplify_label(l) for l in train["labels"]]
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(
        train["embed_pool"],
        labels,
    )
    unique_labels = sorted(set(labels))

    resample_preds = []
    slide_names = []
    for i in range(resample_times):
        val_pkl_p = Path("data/resample/val-{}_pool.pkl".format(i))
        test = pkl_load(val_pkl_p)
        pred = knn.predict(test["embed_pool"])
        resample_preds.append(pred)
        if i == 0:
            slide_names = test["index"]
    y_true = [simplify_label(l) for l in test["labels"]]
    resample_preds_arr = np.array(resample_preds)
    entropy_list = []
    vote_preds = []
    for i in range(len(resample_preds_arr[0])):
        count = Counter(resample_preds_arr[:, i])
        portion = np.array(list(count.values())) / resample_times
        entropy_list.append(entropy(portion))
        vote_preds.append(count.most_common(1)[0][0])
    print("vote_preds")
    dump_metric(y_true, vote_preds, unique_labels, mark="vote_preds")

    is_correct = np.array(vote_preds) == np.array(y_true)
    df = pd.DataFrame(resample_preds_arr, columns=y_true)
    df.to_csv("data/resample/resample_preds.csv")
    df_pred = pd.DataFrame(
        dict(
            slide_names=slide_names,
            is_correct=is_correct,
            entropy=entropy_list,
            pred=vote_preds,
            y_true=y_true,
        ),
    )
    df_pred.to_csv("data/resample/resample_preds_stat.csv")
    false_pred = df_pred.loc[df_pred["is_correct"] == False]
    correct_pred = df_pred.loc[df_pred["is_correct"] == True]
    print(f"false_entropy_mean:{false_pred['entropy'].mean()}, size: {len(false_pred)}")
    print(
        f"correct_entropy_mean:{correct_pred['entropy'].mean()}, size: {len(correct_pred)}",
    )

    low_entropy_pred = df_pred.loc[df_pred["entropy"] < 0.5]
    print("low_entropy_pred size:", len(low_entropy_pred))
    dump_metric(
        low_entropy_pred["y_true"],
        low_entropy_pred["pred"],
        unique_labels,
        mark="low_entropy_pred",
    )


if __name__ == "__main__":
    resample_times = 16
    resampler(resample_times)
    embed_diff(resample_times)
