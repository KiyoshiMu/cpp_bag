from __future__ import annotations

import csv
import math

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

from cpp_bag.io_utils import pkl_load
from cpp_bag.io_utils import simplify_label


def load_size(fp="data/slide_size.csv"):
    with open(fp, "r") as f:
        reader = csv.reader(f)
        next(reader)
        return {row[0]: int(row[1]) for row in reader}


def performance_measure(train_pkl_p, val_pkl_p, mark="pool", random_base=False):
    train = pkl_load(train_pkl_p)
    test = pkl_load(val_pkl_p)
    labels = [simplify_label(l) for l in train["labels"]]
    unique_labels = sorted(set(labels))
    refer_embed = train["embed_pool"]
    n_neighbors = round(math.sqrt(len(refer_embed)))
    print(f"n_neighbors: {n_neighbors}")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance").fit(
        refer_embed,
        labels,
    )
    print(refer_embed.shape)
    y_pred = knn.predict(test["embed_pool"])
    y_true = [simplify_label(l) for l in test["labels"]]
    dump_metric(y_true, y_pred, unique_labels, mark=mark)

    if random_base:
        dummy = DummyClassifier(strategy="stratified", random_state=42).fit(
            refer_embed,
            labels,
        )
        y_pred = dummy.predict(test["embed_pool"])
        dump_metric(y_true, y_pred, unique_labels, mark="dummy")


def dump_metric(y_true, y_pred, unique_labels, mark="pool", to_csv=True):
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=unique_labels,
    )
    print(precision, recall, fscore)
    if to_csv:
        metric_df = pd.DataFrame(
            dict(precision=precision, recall=recall, fscore=fscore),
            index=unique_labels,
        )

        metric_df.to_csv(f"data/{mark}_metric.csv")


if __name__ == "__main__":
    performance_measure(
        "data/train_embed_pool.pkl",
        "data/val_embed_pool.pkl",
    )
