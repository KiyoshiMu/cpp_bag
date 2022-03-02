from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.manifold import TSNE

from cpp_bag.io_utils import simplify_label

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
