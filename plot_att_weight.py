from itertools import chain
from math import floor
import math
from typing import Callable, List
import zipfile
from functools import partial
import numpy as np
from cpp_bag.model import BagPooling
from cpp_bag import data
from pathlib import Path
import torch
import json
import umap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageOps, ImageDraw, ImageFont
from collections import defaultdict

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


MODEL_P = "experiments2/trial0/pool-1669053060376.pth"
DST_DIR = Path("att_rank2")
SPLIT_P = "experiments2/trial0/split0.json"
PRED_P = "experiments2/trial0/pool0.json"

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

    def make_att_df(self, cell_bag:List[data.CellInstance]):
        feature = [cell.feature for cell in cell_bag]
        feature_bag = torch.as_tensor(feature, dtype=torch.float32)
        embedding = self.reducer.fit_transform(feature)
        df = pd.DataFrame(embedding)
        df.columns = ["D1", "D2"]
        df["cellType"] =  [cell.label for cell in cell_bag]
        attn_output_weights = self._attention_weight(feature_bag)
        df["weight"] = attn_output_weights
        df["name"] = [cell.name for cell in cell_bag]
        return df
    
    def make_att_weight(self, cell_bag:List[data.CellInstance]):
        feature = [cell.feature for cell in cell_bag]
        feature_bag = torch.as_tensor(feature, dtype=torch.float32)
        attn_output_weights = self._attention_weight(feature_bag)
        return attn_output_weights
    
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
        cell_threshold=256,
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
    DST_DIR.mkdir(exist_ok=True)
    att_plotter = AttentionWeightPlotter()
    zip_ref = zipfile.ZipFile(Path("D:/DATA/cbp_cell_images.zip"), "r")
    # open image from zip file
    # img_loader = lambda x: Image.open(zip_ref.open(x))
    for index in val_indices:
        _, label, sample_cells = dataset.example_samples(index)
        attn_weight = att_plotter.make_att_weight(sample_cells)

        plot_cell_att_rank(sample_cells, attn_weight, partial(img_loader, zip_ref=zip_ref), marker=label)
        # df = att_plotter.make_att_df(sample_cells, feature)
        # att_plotter.plot_on_df(df, Path("data/att"), img_loader, marker=label)

    zip_ref.close()

def img_loader(x, zip_ref):
    try:
        return Image.open(zip_ref.open(x))
    except KeyError:
        return Image.open(Path("D:/DATA") / x)
    

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
    # main()
    add_pred_info(DST_DIR, PRED_P)