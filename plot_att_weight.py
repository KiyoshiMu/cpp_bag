from itertools import chain
from math import floor
import math
from typing import Callable, List
import zipfile

import numpy as np
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
from PIL import Image, ImageOps, ImageDraw, ImageFont
import re
import plotly.colors
from collections import Counter, defaultdict

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


MODEL_P = "experiments0/trial2/pool-1659028139226.pth"
DST_DIR = Path("att_rank0")
SPLIT_P = "experiments0/trial2/split2.json"
PRED_P = "experiments0/trial2/pool2.json"

def bin_str(bins):
    out = []
    for idx, cut in enumerate(bins[:-1]):
        out.append(f"{cut:.4f}-{bins[idx+1]:.4f}")
    return out[::-1]

def sort_locs_by_type_attW_sum(locs:List[int], types: List[str], att_weights: List[float]):
    sorted_type_count = sort_locs_groups_by_attW_sum(locs, types, att_weights)
    out = list(chain.from_iterable(x[1] for x in sorted_type_count))
    return out

def sort_locs_groups_by_attW_sum(locs:List[int], types: List[str], att_weights: List[float]):
    type_count: defaultdict[str, List[int]] = defaultdict(list)
    for type_, loc in zip(types, locs):
        type_count[type_].append(loc)
    for locs in type_count.values():
        locs.sort(key=lambda loc: att_weights[loc], reverse=True)
    sorted_type_count_pair = sorted(type_count.items(), key=lambda x: sum(att_weights[loc] for loc in x[1]), reverse=True)
    return sorted_type_count_pair

def plot_cell_att_rank_old(sample_cells,rank_map, attn_output_weights,img_loader, bin_strs, marker):
    rank_font_size = 24
    att_font_size = 18
    rank_font = ImageFont.truetype("arial.ttf", rank_font_size)
    att_font = ImageFont.truetype("arial.ttf", att_font_size)
    legend_font = ImageFont.truetype("arial.ttf", 18)
    cell_w = 96
    padding = 20
    left_text_w = 135
    text_padding = 4
    legend_dot_size = 20
    legend_gap = 36
    legend_right_w = 250
    cell_size = cell_w + padding
    # canvas_w = cell_size * max(len(locs) for locs in rank_map.values()) + left_text_w
    # canvas_h = cell_size * 10 + bottom_legend_h
    canvas_w = cell_size * max(len(locs) for locs in rank_map.values()) + left_text_w + legend_right_w
    canvas_h = cell_size * 10
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    slide_name = sample_cells[0].name.split(".")[0]
    for row_idx, rank in enumerate( range(10, 0, -1)):
        locs = rank_map[rank]
        cell_types = [sample_cells[loc].label for loc in locs]
        # locs.sort(key=lambda x:attn_output_weights[x], reverse=True)
        locs = sort_locs_by_type_attW_sum(locs, cell_types, attn_output_weights)
        for col_idx, loc in enumerate(locs):
            cell = sample_cells[loc]
            cell_name = cell.name
            color = COLOR_MAP[cell.label]
            cell_img =img_loader( f"{slide_name}/{cell_name}.jpg")
            cell_img = ImageOps.fit(cell_img, (cell_w, cell_w), method=Image.ANTIALIAS)
            cell_img = ImageOps.expand(cell_img,border=8,fill=color)
            canvas.paste(cell_img, (col_idx * cell_size + left_text_w,  row_idx * cell_size, ))

    for row_idx, rank in enumerate( range(1, 11, 1)):
        d = ImageDraw.Draw(canvas)
        d.text((text_padding, row_idx * cell_size), str(rank), fill=(0, 0, 0), font=rank_font)
        d.text((text_padding, row_idx * cell_size + rank_font_size + 4), bin_strs[row_idx], fill=(0, 0, 0), font=att_font)
    # add legend dots to the bottom
    # start = text_padding
    # for idx, label in enumerate(CELL_TYPES):
    #     color = COLOR_MAP[label]
    #     d = ImageDraw.Draw(canvas)
    #     dot_place = (start, canvas_h - bottom_legend_h, start + legend_dot_size, canvas_h - bottom_legend_h + legend_dot_size)
    #     d.rectangle(dot_place, fill=color)        
    #     d.text((start, canvas_h - bottom_legend_h + legend_dot_size + text_padding), label, fill=(0, 0, 0), font=legend_font)
    #     start = start + legend_dot_size + legend_gap
    start_w = canvas_w - legend_right_w + text_padding * 4
    start_h = text_padding
    for idx, label in enumerate(CELL_TYPES):
        color = COLOR_MAP[label]
        d = ImageDraw.Draw(canvas)
        dot_place = (start_w, start_h, start_w + legend_dot_size, start_h + legend_dot_size)
        d.rectangle(dot_place, fill=color)        
        d.text((start_w + legend_dot_size + text_padding, start_h), label, fill=(0, 0, 0), font=legend_font)
        start_h += legend_gap
    
    
    canvas.save(DST_DIR / f"{slide_name}-{marker}.png")
    
def plot_cell_att_rank(sample_cells:List[data.CellInstance], attn_output_weights,img_loader, marker):
    rank_font_size = 24
    att_font_size = 18
    rank_font = ImageFont.truetype("arial.ttf", rank_font_size)
    att_font = ImageFont.truetype("arial.ttf", att_font_size)
    legend_font = ImageFont.truetype("arial.ttf", 18)
    cell_w = 72
    border = 6
    padding = 16
    left_text_w = 135
    text_padding = 4
    legend_dot_size = 20
    legend_gap = 36
    legend_right_w = 250
    cell_size = cell_w + padding
    slide_name = sample_cells[0].name.split(".")[0]
    sorted_type_count_pair = sort_locs_groups_by_attW_sum([idx for idx, _ in enumerate(sample_cells)],
                                                        [c.label for c in sample_cells],
                                                        attn_output_weights)
    ROW_CELL_N = 20
    row_count = 0
    for _, locs in sorted_type_count_pair:
        row_count += math.ceil(len(locs) / ROW_CELL_N)

    canvas_w = cell_size * ROW_CELL_N + left_text_w + legend_right_w
    canvas_h = cell_size * row_count
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    row_idx = 0
    cell_type_row_idx = {}
    for _type, locs in sorted_type_count_pair:
        cell_type_row_idx[_type] = row_idx
        for col_idx, loc in enumerate(locs):
            cell = sample_cells[loc]
            cell_name = cell.name
            color = COLOR_MAP[cell.label]
            cell_img =img_loader( f"{slide_name}/{cell_name}.jpg")
            cell_img = ImageOps.fit(cell_img, (cell_w, cell_w), method=Image.ANTIALIAS)
            cell_img = ImageOps.expand(cell_img,border=border,fill=color)
            col_loc = col_idx % ROW_CELL_N
            row_loc = row_idx + col_idx // ROW_CELL_N
            canvas.paste(cell_img, (col_loc * cell_size + left_text_w,  row_loc * cell_size, ))
        row_idx = row_loc + 1
    
    d = ImageDraw.Draw(canvas)
    for rank, (_type, locs) in enumerate(sorted_type_count_pair, start=1):
        type_idx = cell_type_row_idx[_type]
        avg_att = sum(attn_output_weights[loc] for loc in locs) / len(locs)
        d.text((text_padding, type_idx * cell_size), str(rank), fill=(0, 0, 0), font=rank_font)
        d.text((text_padding, type_idx * cell_size + rank_font_size + 4), _type, fill=(0, 0, 0), font=att_font)
        d.text((text_padding, type_idx * cell_size + rank_font_size + att_font_size + 8), f"{avg_att:.4f}", fill=(0, 0, 0), font=att_font)
    
    start_w = canvas_w - legend_right_w + text_padding * 4
    start_h = text_padding
    for cell_type in cell_type_row_idx.keys():
        color = COLOR_MAP[cell_type]
        d = ImageDraw.Draw(canvas)
        dot_place = (start_w, start_h, start_w + legend_dot_size, start_h + legend_dot_size)
        d.rectangle(dot_place, fill=color)        
        d.text((start_w + legend_dot_size + text_padding, start_h), cell_type, fill=(0, 0, 0), font=legend_font)
        start_h += legend_gap
    
    
    canvas.save(DST_DIR / f"{slide_name}-{marker}.png")
    
def mk_color_map(cell_types):
    colors = px.colors.qualitative.Alphabet
    color_map = {t: colors[i % len(colors)] for i, t in enumerate(cell_types)}
    return color_map

COLOR_MAP = mk_color_map(CELL_TYPES)
WIDTH = 940
HEIGHT = 600
WHITE_CANVAS = Image.new("RGB", (WIDTH, HEIGHT), color=(255, 255, 255))
class AttentionWeightPlotter:
    def __init__(self, mp = MODEL_P):
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
    
    def make_att_weight(self, cell_bag:List[data.CellInstance], feature_bag: torch.Tensor):
        attn_output_weights = self._attention_weight(feature_bag)[:len(cell_bag)]
        return attn_output_weights
    
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
            cell_img.putalpha(floor(size*255))
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
                    opacity=1,
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

def get_rank_map(attn_output_weights, bins):
    ranks = np.digitize(attn_output_weights, bins, right=True)
    rank_map = defaultdict(list)
    for idx, rank in enumerate(ranks):
        rank_map[rank].append(idx)
    return rank_map
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
        enable_mask=True
    )
    size = len(dataset)
    print("size:", size)
    
    with open(SPLIT_P, "r") as f:
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
        attn_weight = att_plotter.make_att_weight(sample_cells, feature)
        # _, bins = pd.qcut(attn_weight, 10, retbins=True)
        # bin_strs = bin_str(bins)
        # rank_map = get_rank_map(attn_weight, bins)
        # plot_cell_att_rank_old(sample_cells, rank_map, attn_weight, img_loader, bin_strs, marker=label)

        plot_cell_att_rank(sample_cells, attn_weight, img_loader, marker=label)
        # df = att_plotter.make_att_df(sample_cells, feature)
        # att_plotter.plot_on_df(df, Path("data/att"), img_loader, marker=label)

    zip_ref.close()

def add_pred_info(att_dir:Path, ret_p):
    with open(ret_p, "r") as f:
        preds = json.load(f)
    pred_map = {p["index"]: (p["label"], p["pred"]) for p in preds}
    att_dir_map = {f.stem.split("-")[0]: f for f in att_dir.glob("*.png")}
    for k, v in att_dir_map.items():
        label, pred = pred_map[k]
        att_label = v.stem.split("-")[1]
        assert label in att_label
        v.rename(v.with_name(f"{k}-{label}-{pred}.png"))
    


if __name__ == "__main__":
    main()
    # add_pred_info(DST_DIR, PRED_P)