from __future__ import annotations

from pathlib import Path


files = [
    f.stem
    for f in Path(
        "/storage/create_local/campbell/Campbell_Lab/Research/dzi_merged/",
    ).glob("*.json")
    if f.stat().st_size < 10
]

print(files)
