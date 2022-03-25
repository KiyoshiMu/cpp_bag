from __future__ import annotations

import pandas as pd
import plotly.express as px

TEMPLATE = "plotly_white"
FONT = "Arial"


def plot_entropy():
    df = pd.read_csv("data\\resample\\resample_preds_stat.csv")
    fig = px.histogram(
        df,
        x="entropy",
        color="is_correct",
        hover_data=["pred", "y_true"],
    )
    fig.update_layout(template=TEMPLATE, font_family="Arial", barmode="overlay")
    fig.write_image("plot_entropy.jpeg", scale=3)


if __name__ == "__main__":
    plot_entropy()
