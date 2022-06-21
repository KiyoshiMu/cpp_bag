from typing import Callable, List
import zipfile
from cpp_bag.model import BagPooling
from cpp_bag import data
from pathlib import Path
import torch
import json
from sklearn.preprocessing import minmax_scale
import umap
import pandas as pd
import plotly.express as px
from hflayers import  HopfieldPooling
import plotly.graph_objects as go
from PIL import Image, ImageOps
import re
import plotly.colors

TEMPLATE = "plotly_white"
FONT = "Arial"

class AttentionWeightPlotter:
    def __init__(self, mp = "pool-1648142022566_MK.pth"):
        self.reducer = umap.UMAP()
        in_dim = 256
        model = BagPooling.from_checkpoint(mp, in_dim=in_dim)
        self.pooling = model.pooling

    def make_att_df(self, cell_bag:List[data.CellInstance], feature_bag: torch.Tensor):
        feature = [cell.feature for cell in cell_bag]
        embedding = self.reducer.fit_transform(feature)
        df = pd.DataFrame(embedding)
        df.columns = ["D1", "D2"]
        df["cellType"] =  [cell.label for cell in cell_bag]
        attn_output_weights = self._attention_weight(feature_bag)[:len(cell_bag)]
        df["weight"] = minmax_scale(attn_output_weights, (0, 1))
        df["size"] = minmax_scale(attn_output_weights, (0.1, 1))
        df["name"] = [cell.name for cell in cell_bag]
        return df
    
    def plot_on_df(self, df: pd.DataFrame, save_path: Path , img_loader: Callable[[str], Image.Image], marker=None ):
        fig = go.Figure()
        fig.add_trace(
                    go.Scatter(
                        mode='markers',
                        x=df["D1"],
                        y=df["D2"],
                        marker=dict(
                            opacity=0,
                        ),
                        showlegend=False
                    )
                )
        slide_name = df.iloc[0]["name"].split(".")[0]
        for i, row in df.iterrows():
            weight = row["weight"]
            size = row["size"]
            cell_img =img_loader( f"{slide_name}/{row['name']}.jpg")
            color = get_continuous_color(plotly.colors.PLOTLY_SCALES["RdBu"], weight)
            cell_img = ImageOps.expand(cell_img,border=6,fill=color)
            fig.add_layout_image(
                dict(
                    source=cell_img,
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=row["D1"],
                    y=row["D2"],
                    sizex=0.35,
                    sizey=0.35,
                    sizing="contain",
                    opacity=size,
                    layer="above"
                )
            )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            template=TEMPLATE,
            font_family=FONT,
            legend=dict(
                orientation="h",
            ),
            width=940,
            height=600,
            )
        if marker is not None:
            # fig.write_image(save_path/ f"{slide_name}-{marker}.att.pdf", format="pdf")
            fig.write_image(save_path/ f"{slide_name}-{marker}.att.jpg", format="jpg", scale=3)
        else:
            fig.write_image(save_path/ f"{slide_name}.att.jpg", format="jpg",scale=3)
        
        
    def _attention_weight(self, feature: torch.Tensor):
        input_ = feature.unsqueeze(dim=0)
        attn_raw = self.pooling.get_association_matrix(input_)
        num_heads = attn_raw.size()[1]
        # average attention weights over heads
        attn_output_weights = attn_raw.sum(dim=1) / num_heads
        return attn_output_weights.squeeze()
    
def main():
    WITH_MK = True
    all_cells = data.load_cells()
    dataset = data.CustomImageDataset(
        data.FEAT_DIR,
        data.LABEL_DIR,
        bag_size=256,
        cell_threshold=300,
        with_MK=WITH_MK,
        all_cells=all_cells,
    )
    size = len(dataset)
    print("size:", size)
    
    with open("data/split.json", "r") as f:
        cache = json.load(f)
        val_indices = cache["val"]
        # train_indices = cache["train"]
    # val_set = data.Subset(dataset, val_indices)
    # train_set = data.Subset(dataset, train_indices)
    
    att_plotter = AttentionWeightPlotter()
    zip_ref = zipfile.ZipFile(Path("D:/DATA/cbp_cell_images.zip"), "r")
    # open image from zip file
    img_loader = lambda x: Image.open(zip_ref.open(x))
    for index in val_indices:
        feature, label, sample_cells = dataset.example_samples(index)
        df = att_plotter.make_att_df(sample_cells, feature)
        att_plotter.plot_on_df(df, Path("data/att"), img_loader, marker=label)


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    color_str = plotly.colors.find_intermediate_color(
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")
    return tuple(int(float(f)) for f in re.findall(r'\d+.\d+', color_str))


if __name__ == "__main__":
    main()