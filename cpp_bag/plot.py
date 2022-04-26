from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.dummy import DummyClassifier
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from cpp_bag.io_utils import json_dump
from cpp_bag.io_utils import pkl_load
from cpp_bag.io_utils import simplify_label
from cpp_bag.performance import load_size

TEMPLATE = "plotly_white"
FONT = "Arial"


def arr_project(arrays, method="umap"):
    if method == "umap":
        reducer = umap.UMAP(n_components=2)
    else:
        reducer = TSNE(n_components=2)
    return reducer.fit_transform(arrays)


def make_projection(embed, labels, index, source):

    projection = arr_project(embed)
    plot_df = pd.DataFrame(
        {
            "D1": projection[:, 0],
            "D2": projection[:, 1],
            "label": labels,
            "index": index,
            "source": source,
        },
    )
    plot_df.to_json("slide_embed.json")


def avg_pool(arr):
    arr_norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr_normalized = arr / arr_norm
    arr_avg = arr_normalized.mean(axis=0)
    return arr_avg


def make_plot(embed, labels, index, mark="", write=True, dst=Path(".")):

    projection = arr_project(embed)
    plot_df = pd.DataFrame(
        {
            "D1": projection[:, 0],
            "D2": projection[:, 1],
            "label": labels,
            "index": index,
        },
    )
    fig = state_plot(plot_df, thresh=10, use_simplify=True)
    if write:
        plot_df.to_csv(str(dst / f"{mark}plot_df.csv"))
        fig.write_html(str(dst / f"{mark}umap.html"))
    return fig


def state_plot(df: pd.DataFrame, thresh=15, use_simplify=False):
    if use_simplify:
        df["marker_base"] = df["label"].map(simplify_label)
    else:
        df["marker_base"] = df["label"]
    markers = df["marker_base"].value_counts()
    label_value_counts = markers >= thresh
    show_legends = set(label_value_counts.index.to_numpy()[label_value_counts])
    df["show_legends"] = df["marker_base"].map(lambda x: x in show_legends)

    raw_symbols = SymbolValidator().values
    colors = px.colors.qualitative.Plotly
    symbol_dict = {}
    color_dict = {}
    color_len = len(colors)
    for idx, label in enumerate(markers.index):
        symbol_idx = idx // color_len
        color_idx = idx % color_len
        symbol_dict[label] = raw_symbols[symbol_idx]
        color_dict[label] = colors[color_idx]
    df["color"] = df["marker_base"].map(color_dict)
    df["symbol"] = df["marker_base"].map(symbol_dict)

    fig = go.Figure()
    sel_labels = sorted(show_legends, key=len)
    for sel_label in sel_labels:
        tmp_df = df.loc[df["marker_base"] == sel_label, :]
        fig.add_trace(
            go.Scatter(
                x=tmp_df["D1"],
                y=tmp_df["D2"],
                mode="markers",
                marker_color=tmp_df["color"],
                marker_symbol=tmp_df["symbol"],
                text=tmp_df["label"].to_list(),
                showlegend=True,
                name=sel_label,
            ),
        )

    no_legend_df = df.loc[~df["show_legends"]]
    fig.add_trace(
        go.Scatter(
            x=no_legend_df["D1"],
            y=no_legend_df["D2"],
            mode="markers",
            opacity=0.5,
            marker_color=no_legend_df["color"],
            marker_symbol=no_legend_df["symbol"],
            showlegend=False,
            text=no_legend_df["label"].to_list(),
            name="",
        ),
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
    return fig


def slide_vectors(
    train_pkl_p,
    val_pkl_p,
    val_entropy: Optional[list] = None,
    mark="pool",
    dst=Path("."),
):
    train = pkl_load(train_pkl_p)
    refer_embed = train["embed_pool"]
    labels = [simplify_label(l) for l in train["labels"]]
    n_neighbors = round(math.sqrt(len(refer_embed)))
    print(f"n_neighbors: {n_neighbors}")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance").fit(
        refer_embed,
        labels,
    )
    classes_ = knn.classes_
    val = pkl_load(val_pkl_p)
    val_embed = val["embed_pool"]
    projection = arr_project(val_embed)
    pred_probs: np.ndarray = knn.predict_proba(val_embed)
    val_labels = [simplify_label(l) for l in val["labels"]]
    _df = proba_to_df(pred_probs, classes_, val_labels)
    preds = knn.predict(val_embed)
    sizes = load_size()
    val_names = val["index"]
    _df.update(
        {
            "D1": projection[:, 0],
            "D2": projection[:, 1],
            "label": val_labels,
            "full_label": val["labels"],
            "index": val_names,
            "cell_count": [sizes[n] for n in val_names],
            "pred": preds,
            "correct": [
                val_labels == preds for (val_labels, preds) in zip(val_labels, preds)
            ],
        },
    )
    if val_entropy is not None:
        _df["pred_entropy"] = val_entropy

    plot_df = pd.DataFrame(_df)
    summary = top3_summary(plot_df)
    dummy_summary = dummy_exp(refer_embed, labels, val_embed, val_labels)
    print("summary", summary)
    print("dummy_summary", dummy_summary)
    json_dump(summary, dst / f"{mark}_slide_summary.json")
    json_dump(dummy_summary, dst / f"{mark}_slide_dummy_summary.json")
    plot_df.to_json(str(dst / f"{mark}.json"), orient="records")
    fig = px.scatter(
        plot_df,
        x="D1",
        y="D2",
        color="label",
        hover_name="index",
        symbol="correct",
        hover_data=[
            "pred",
            "prob_top0",
            "prob_top1",
            "prob_top2",
            "full_label",
            "cell_count",
        ],
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


def proba_to_df(pred_probs, classes_, val_labels):
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
    top3_corrects = [
        any(
            e
            for e in (prob_top0[idx], prob_top1[idx], prob_top2[idx])
            if ("0.00" not in e and val_labels[idx] in e)
        )
        for idx in range(len(val_labels))
    ]
    _df = {
        "label": val_labels,
        "prob_top0": prob_top0,
        "prob_top1": prob_top1,
        "prob_top2": prob_top2,
        "top3_correct": top3_corrects,
    }
    return _df


def top3_summary(cases):
    correct_cases = cases[cases["top3_correct"]]
    incorrect_cases = cases[~cases["top3_correct"]]
    summary = {
        "correct": (len(correct_cases), len(correct_cases) / len(cases)),
        "incorrect": (len(incorrect_cases), len(incorrect_cases) / len(cases)),
    }
    return summary


def dummy_exp(refer_embed, refer_labels, test_embed, test_labels):
    dummy = DummyClassifier(strategy="stratified", random_state=42).fit(
        refer_embed,
        refer_labels,
    )
    classes_ = dummy.classes_
    pred_probs = dummy.predict_proba(test_embed)
    _df = proba_to_df(pred_probs, classes_, test_labels)
    return top3_summary(pd.DataFrame(_df))
