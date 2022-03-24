from __future__ import annotations

import json

import pandas as pd

# from cpp_bag import data

# def probe():
#     dataset = data.CustomImageDataset(data.FEAT_DIR, data.LABEL_DIR, 256)
#     slideNames = dataset.slide_names
#     cellSizes = [f.size(dim=0) for f in dataset.features]
#     df = pd.DataFrame({"slide": slideNames, "cell_size": cellSizes})
#     df.to_csv("data/slide_size.csv", index=False)


def check():
    ref = pd.read_csv("data/slide_size.csv")
    df = pd.read_csv("data/metaA.csv")
    meet_req = [f"{n}.tif" for n in ref.loc[ref["cell_size"] >= 1000, "slide"]]
    extra_df = df.loc[
        (df["tiff.XResolution"] == 50000) & (df["filename"].str.contains("AS"))
    ].loc[:, ["filename", "tiff.XResolution"]]
    extra_df = extra_df.loc[~extra_df["filename"].isin(meet_req)]
    extra_df.to_csv(
        "data/extra.csv",
        index=False,
    )
    with open("data/extra.json", "w") as f:
        json.dump([n.replace(".tif", "") for n in extra_df["filename"]], f)


def list_names():
    df = pd.read_csv("data/metaA.csv")
    names = df.loc[
        (df["tiff.XResolution"] == 50000) & (df["filename"].str.contains("AS"))
    ]["filename"].to_list()
    with open("data/slide_names.json", "w") as f:
        json.dump([n.replace(".tif", "") for n in names], f)


if __name__ == "__main__":
    check()
