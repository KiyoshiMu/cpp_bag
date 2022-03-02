from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

from cpp_bag.io_utils import pkl_load
from cpp_bag.io_utils import simplify_label


def performance(train_pkl_p, val_pkl_p, mark="pool", random_base=False):
    train = pkl_load(train_pkl_p)
    test = pkl_load(val_pkl_p)
    labels = [simplify_label(l) for l in train["labels"]]
    unique_labels = sorted(set(labels))
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(
        train["embed_pool"],
        labels,
    )
    print(train["embed_pool"].shape)
    y_pred = knn.predict(test["embed_pool"])
    y_true = [simplify_label(l) for l in test["labels"]]
    dump_metric(y_true, y_pred, unique_labels, mark=mark)

    if random_base:
        dummy = DummyClassifier(strategy="stratified", random_state=42).fit(
            train["embed_pool"],
            labels,
        )
        y_pred = dummy.predict(test["embed_pool"])
        print("dummy", y_pred)
        dump_metric(y_true, y_pred, unique_labels, mark="dummy")


def dump_metric(y_true, y_pred, labels, mark="pool"):
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
    )
    print(precision, recall, fscore)
    metric_df = pd.DataFrame(
        dict(precision=precision, recall=recall, fscore=fscore),
        index=labels,
    )
    metric_df.to_csv(f"data/{mark}_metric.csv")


if __name__ == "__main__":
    performance(
        "data/train_embed_pool.pkl",
        "data/val_embed_pool.pkl",
    )
