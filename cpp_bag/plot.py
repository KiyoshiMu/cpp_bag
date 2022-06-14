from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import umap
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.manifold import TSNE

from cpp_bag.io_utils import json_dump
from cpp_bag.io_utils import pkl_load
from cpp_bag.io_utils import simplify_label
from cpp_bag.performance import create_knn
from cpp_bag.performance import dummy_exp
from cpp_bag.performance import dump_metric
from cpp_bag.performance import load_size
from cpp_bag.performance import proba_to_dfDict
from cpp_bag.performance import top3_summary

TEMPLATE = "plotly_white"
FONT = "Arial"


def heat_map(z, x, y, annotation_text):
    fig = ff.create_annotated_heatmap(
        z,
        y=y,
        x=x,
        annotation_text=annotation_text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        width=1280,
        height=600,
        yaxis_visible=False,
        yaxis_showticklabels=False,
    )
    return fig


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
    knn = create_knn(refer_embed, labels)
    classes_ = knn.classes_
    val = pkl_load(val_pkl_p)
    val_embed = val["embed_pool"]
    projection = arr_project(val_embed)
    pred_probs: np.ndarray = knn.predict_proba(val_embed)
    val_labels = [simplify_label(l) for l in val["labels"]]
    _df = proba_to_dfDict(pred_probs, classes_, val_labels)
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
    dump_metric(val_labels, preds, classes_, dst / f"{mark}_metric.csv")
    dummy_summary = dummy_exp(
        refer_embed,
        labels,
        val_embed,
        val_labels,
        dst / f"{mark}_dummy_metric.csv",
    )
    # print("summary", summary)
    # print("dummy_summary", dummy_summary)
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


def plot_tag_perf_with_std(performance, main_metrics="F1 Score", include_random=False):
    #  perf_average, perf_err

    performance.sort_values(
        f"{main_metrics}_mean",
        inplace=True,
    )
    fig = go.Figure()
    x = performance.index.to_list()
    marker_symbols = ["square", "x"]
    # fig.add_shape(
    #     type="line",
    #     x0=0.01,
    #     y0=perf_average,
    #     x1=0.99,
    #     y1=perf_average,
    #     xref="paper",
    #     line=dict(color="lightgray", width=2, dash="dash"),
    # )
    for idx, measure in enumerate(["Precision", "Recall"]):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=performance[f"{measure}_mean"],
                error_y=dict(
                    color="lightgray",
                    type="data",
                    array=performance[f"{measure}_std"],
                    visible=False,
                ),
                marker_color="lightgray",
                marker_symbol=marker_symbols[idx],
                marker_size=10,
                mode="markers",
                name=measure,
            ),
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=performance[f"{main_metrics}_mean"],
            error_y=dict(
                color="lightgray",
                type="data",
                array=performance[f"{main_metrics}_std"],
                visible=True,
            ),
            marker_color="blue",
            mode="markers+text",
            text=[f"{v:.02f}" for v in performance[f"{main_metrics}_mean"]],
            marker_size=10,
            name=main_metrics,
        ),
    )
    if include_random:
        random_title = "Random"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=performance[f"{random_title}_mean"],
                error_y=dict(
                    color="lightgray",
                    type="data",
                    array=performance[f"{random_title}_std"],
                    visible=False,
                ),
                marker_color="pink",
                mode="markers+text",
                text=[f"{v:.02f}" for v in performance[f"{random_title}_mean"]],
                marker_size=10,
                name=f"{random_title} {main_metrics}",
            ),
        )
    fig.update_traces(textposition="middle right")

    x_loc = len(performance) - 2
    fig.update_layout(
        template=TEMPLATE,
        font_family="Arial",
        width=1280,
        height=600,
        xaxis_title="Label",
        yaxis_title="Metrics",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # annotations=[
        #     dict(
        #         x=x_loc,
        #         y=perf_average,
        #         xref="x",
        #         yref="y",
        #         text=f"Micro {y_title}: {perf_average:.03f}±{perf_err:.03f}",
        #         showarrow=False,
        #         font=dict(size=15),
        #     ),
        # ],
    )
    fig.add_annotation(
        x=1,
        y=1.05,
        xref="paper",
        yref="paper",
        align="left",
        text="n=5 independent experiments",
        showarrow=False,
    )
    return fig
