from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

from cpp_bag.io_utils import pkl_load
from cpp_bag.io_utils import simplify_label
from cpp_bag.plot import arr_project

TEMPLATE = "plotly_white"
FONT = "Arial"


def graph(train_pkl_p, val_pkl_p, mark="pool", dst=Path(".")):
    train = pkl_load(train_pkl_p)
    refer_embed = train["embed_pool"]
    labels = [simplify_label(l) for l in train["labels"]]
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(
        refer_embed,
        labels,
    )
    classes_ = knn.classes_
    val = pkl_load(val_pkl_p)
    val_embed = val["embed_pool"]
    projection = arr_project(val_embed)
    pred_probs: np.ndarray = knn.predict_proba(val_embed)
    pred_probs_argsort = np.argsort(pred_probs, axis=1)[:, ::-1]
    prob_top0 = [
        f"{classes_[indices[0]]}:{pred_probs[row_idx, indices[0]]:.2f}"
        for row_idx, indices in enumerate(pred_probs_argsort)
    ]
    prob_top1 = [
        f"{classes_[indices[1]]}:{pred_probs[row_idx, indices[1]]:.2f}"
        for row_idx, indices in enumerate(pred_probs_argsort)
    ]
    prob_top2 = [
        f"{classes_[indices[2]]}:{pred_probs[row_idx, indices[2]]:.2f}"
        for row_idx, indices in enumerate(pred_probs_argsort)
    ]
    preds = knn.predict(val_embed)
    val_labels = [simplify_label(l) for l in val["labels"]]
    plot_df = pd.DataFrame(
        {
            "D1": projection[:, 0],
            "D2": projection[:, 1],
            "label": val_labels,
            "full_label": val["labels"],
            "index": val["index"],
            "pred": preds,
            "prob_top0": prob_top0,
            "prob_top1": prob_top1,
            "prob_top2": prob_top2,
            "correct": [
                val_labels == preds for (val_labels, preds) in zip(val_labels, preds)
            ],
        },
    )
    fig = px.scatter(
        plot_df,
        x="D1",
        y="D2",
        color="label",
        hover_name="index",
        symbol="correct",
        hover_data=["pred", "prob_top0", "prob_top1", "prob_top2", "full_label"],
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        legend=dict(
            orientation="h",
        ),
        width=1280,
        height=600,
    )
    fig.write_html(str(dst / f"{mark}umap.html"))
    return fig


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
