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
CELL_TYPES: list[str] = [
    "Neutrophil",
    "Metamyelocyte",
    "Myelocyte",
    "Promyelocyte",
    "Blast",
    "Erythroblast",
    "Megakaryocyte_nucleus",
    "Lymphocyte",
    "Monocyte",
    "Plasma_cell",
    "Eosinophil",
    "Basophil",
    "Histiocyte",
    "Mast_cell",
]
def mk_color_map(cell_types):
    colors = px.colors.qualitative.Alphabet
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(cell_types)}
    return color_map

COLOR_MAP = mk_color_map(CELL_TYPES)
WIDTH = 940
HEIGHT = 600
WHITE_CANVAS = Image.new("RGB", (WIDTH, HEIGHT), color=(255, 255, 255))
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
        df["weight"] = minmax_scale(attn_output_weights, (0, 0.999))
        df["size"] = minmax_scale(attn_output_weights, (0.1, 0.999))
        df["name"] = [cell.name for cell in cell_bag]
        return df
    
    def plot_on_df(self, df: pd.DataFrame, save_path: Path , img_loader: Callable[[str], Image.Image], marker=None ):
        fig = go.Figure()
        cell_types = df["cellType"].unique().tolist()
        color_map = COLOR_MAP
        x_min = df["D1"].min()
        x_max = df["D1"].max()
        y_min = df["D2"].min()
        y_max = df["D2"].max()
        for cell_type in cell_types:
            cell_df = df[df["cellType"] == cell_type]
            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=cell_df["D1"],
                    y=cell_df["D2"],
                    marker=dict(
                        opacity=1,
                        size=10,
                        color=color_map[cell_type],
                    ),
                    showlegend=True,
                    name=cell_type,
                )
            )

        # hide dots
        fig.add_layout_image(
            dict(
                source=WHITE_CANVAS,
                xref="x",
                yref="y",
                x=x_min-1,
                y=y_max+1,
                sizex=(x_max-x_min+2),
                sizey=(y_max-y_min+2),
                sizing="stretch",
                opacity=1,
                layer="above")
        )
        
        slide_name = df.iloc[0]["name"].split(".")[0]
        for i, row in df.iterrows():
            # weight = row["weight"]
            size = row["size"]
            cell_img =img_loader( f"{slide_name}/{row['name']}.jpg")
            # color = get_continuous_color(plotly.colors.PLOTLY_SCALES["RdBu"], weight)
            color = color_map[row["cellType"]]
            cell_img = ImageOps.expand(cell_img,border=7,fill=color)
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
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_layout(
            template=TEMPLATE,
            font_family=FONT,
            # legend=dict(
            #     orientation="h",
            # ),
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
        att_plotter.plot_on_df(df, Path("data/att_"), img_loader, marker=label)


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

def add_pred_info(att_dir, ret_p):
    with open(ret_p, "r") as f:
        preds = json.load(f)
    pred_map = {p["index"]: (p["label"], p["pred"]) for p in preds}
    att_dir_map = {f.stem.split("-")[0]: f for f in Path(att_dir).glob("*.att.jpg")}
    for k, v in att_dir_map.items():
        label, pred = pred_map[k]
        att_label = v.stem.split("-")[1]
        assert label in att_label
        v.rename(v.with_name(f"{k}-{label}-{pred}.att.jpg"))
    


if __name__ == "__main__":
    # main()
    add_pred_info("data/att_", "slide_vectors.json")