from __future__ import annotations

from pathlib import Path
from typing import Callable
from typing import Optional

import numpy as np
import torch

from cpp_bag.data import CustomImageDataset
from cpp_bag.io_utils import pkl_dump
from cpp_bag.plot import make_plot


def ds_project(
    ds: CustomImageDataset,
    embed_func: Callable[[CustomImageDataset], np.ndarray],
    dst_dir=Path("."),
    name_mark="val_",
    resample_times=16,
):
    embed_pools = []
    for _ in range(resample_times):
        embed_pools.append(embed_func(ds))
    embed_pool = np.stack(embed_pools, axis=0).mean(axis=0)
    pkl_dst = str(dst_dir / f"{name_mark}_pool.pkl")
    labels, index = ds.labels, ds.slide_names
    pkl_dump(dict(embed_pool=embed_pool, labels=labels, index=index), pkl_dst)
    fig = make_plot(embed_pool, labels, index, mark=name_mark, dst=dst_dir)
    return pkl_dst, fig


def ds_embed(ds: CustomImageDataset, model):
    model.eval()
    dataloder = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    embed: Optional[torch.Tensor] = None
    with torch.no_grad():
        for idx, (X, _) in enumerate(dataloder):
            if idx == 0:
                embed = model(X)
            elif embed is not None:
                embed = torch.cat((embed, model(X)), dim=0)
    return embed.numpy() if embed is not None else None


def ds_avg(ds: CustomImageDataset):
    avg_pool = np.array([feat.mean(dim=0).numpy() for (feat, _) in ds])
    return avg_pool
