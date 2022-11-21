from pathlib import Path
from typing import List
import zipfile
from PIL import Image, ImageOps, ImageDraw, ImageFont
from cpp_bag import data
import plotly.express as px
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
def main():
    WITH_MK = True
    all_cells = data.load_cells()
    size =10
    dataset = data.CustomImageDataset(
        data.FEAT_DIR,
        data.LABEL_DIR,
        bag_size=256,
        cell_threshold=256,
        with_MK=WITH_MK,
        all_cells=all_cells,
        limit=size
    )
    zip_ref = zipfile.ZipFile(Path("D:/DATA/cbp_cell_images.zip"), "r")
    img_loader = lambda x: Image.open(zip_ref.open(x))
    for index in range(size):
        _, _, sample_cells = dataset.example_samples(index)
        plot_sample_cells(sample_cells, img_loader)
        
def plot_sample_cells(sample_cells:List[data.CellInstance], img_loader):
    cell_w = 72
    padding = 16
    border = 6
    ROW_CELL_N = 16
    cell_size = cell_w + padding
    left_text_w = 10
    text_padding = 4
    legend_dot_size = 20
    legend_gap = 36
    legend_right_w = 250
    legend_font = ImageFont.truetype("arial.ttf", 18)
    canvas_w = cell_size * ROW_CELL_N + left_text_w + legend_right_w
    canvas_h = cell_size * ROW_CELL_N
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    slide_name = sample_cells[0].name.split(".")[0]
    sample_cells.sort(key=lambda x: CELL_TYPES.index(x.label))
    for idx, cell in enumerate(sample_cells):
        cell_name = cell.name
        cell_img =img_loader( f"{slide_name}/{cell_name}.jpg")
        cell_img = ImageOps.fit(cell_img, (cell_w, cell_w), method=Image.ANTIALIAS)
        cell_img = ImageOps.expand(cell_img,border=border,fill=COLOR_MAP[cell.label])
        col_idx = idx % ROW_CELL_N
        row_idx = idx // ROW_CELL_N
        canvas.paste(cell_img, (col_idx * cell_size + left_text_w,  row_idx * cell_size, ))
    
    start_w = canvas_w - legend_right_w + text_padding * 4
    start_h = text_padding
    cell_types = set(c.label for c in sample_cells)
    d = ImageDraw.Draw(canvas)
    for _, label in enumerate(CELL_TYPES):
        if label not in cell_types:
            continue
        dot_place = (start_w, start_h, start_w + legend_dot_size, start_h + legend_dot_size)
        d.rectangle(dot_place, fill=COLOR_MAP[label])        
        d.text((start_w + legend_dot_size + text_padding, start_h), label, fill=(0, 0, 0), font=legend_font)
        start_h += legend_gap
        
    canvas.save(f"cell_bag_img/{slide_name}.png")
    
if __name__ == "__main__":
    main()