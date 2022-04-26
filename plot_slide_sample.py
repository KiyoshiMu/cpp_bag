from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

random.seed(42)
Lineage = set(
    """\
Neutrophil,Metamyelocyte,Myelocyte,\
Promyelocyte,Blast,Erythroblast,Megakaryocyte_nucleus,\
Lymphocyte,Monocyte,Plasma_cell,Eosinophil,Basophil,\
Histiocyte,Mast_cell""".split(
        ",",
    ),
)


def load_slide_embed(path):
    with open(path, "r") as f:
        return json.load(f)


def top3_correct(case: dict, label: str) -> bool:
    return any(
        e
        for e in (case["prob_top0"], case["prob_top1"], case["prob_top2"])
        if ("0.00" not in e and label in e)
    )


def slide_sample(cases, sample_size=0.25):
    correct_cases = [case for case in cases if top3_correct(case, case["label"])]
    incorrect_cases = [case for case in cases if not top3_correct(case, case["label"])]
    for case in correct_cases:
        case["top3_correct"] = True
    for case in incorrect_cases:
        case["top3_correct"] = False
    print(f"correct: {len(correct_cases)} {len(correct_cases) / len(cases):.2%}")
    print(f"incorrect: {len(incorrect_cases)} {len(incorrect_cases) / len(cases):.2%}")
    correct_samples = random.sample(
        correct_cases,
        round(len(correct_cases) * sample_size),
    )
    incorrect_samples = random.sample(
        incorrect_cases,
        round(len(incorrect_cases) * sample_size),
    )
    for case in correct_samples:
        case["sampled"] = True
    for case in incorrect_samples:
        case["sampled"] = True
    assert len([case for case in cases if case.get("sampled", False)]) == round(
        len(correct_cases) * sample_size,
    ) + round(len(incorrect_cases) * sample_size)
    return cases


FEAT_DIR = Path("/storage/create_local/campbell/Campbell_Lab/Research/dzi_merged/")
CP_FEAT_DIR = Path("data/sampled_feats")


def main():
    cases = load_slide_embed("slide_vectors.json")
    new_cases = slide_sample(cases)
    with open("slide_vectors_sample.json", "w") as f:
        json.dump(new_cases, f)
    # CP_FEAT_DIR.mkdir(exist_ok=True)
    # cases = load_slide_embed("slide_vectors_sample.json")
    # slide_names = [case["index"] for case in cases if case.get("sampled", False)]
    # for slide_name in slide_names:
    #     slide_feat_p = FEAT_DIR / f"{slide_name}.json"
    #     slide_feat_cp_p = CP_FEAT_DIR / f"{slide_name}.json"
    #     data = load_slide_embed(slide_feat_p)
    #     new_data = [item for item in data if item['label'] in Lineage]
    #     with open(slide_feat_cp_p, "w") as f:
    #         json.dump(new_data, f)


def hci_help(dir_p: str):
    files = [f for f in Path(dir_p).glob("*.csv")]
    names = []
    dst_dir = Path("hci_feat")
    dst_dir.mkdir(exist_ok=True)
    for file in files:
        slide_name = file.stem
        feat_p = FEAT_DIR / f"{slide_name}.json"
        if not feat_p.exists():
            print(f"{slide_name} not found")
            continue
        data = load_slide_embed(feat_p)
        targets = set(pd.read_csv(file)["image_file"])
        new_data = [item for item in data if item["src"] in targets]
        if len(new_data) == 0:
            print(f"{slide_name} no target")
            continue
        feat_cp_p = dst_dir / f"{slide_name}.json"
        with open(feat_cp_p, "w") as f:
            json.dump(new_data, f)
        names.append(slide_name)
    with open("hci_index.json", "w") as f:
        json.dump(names, f)


EXTRA = ["19_0658_AS", "19_0666_AS", "19_0689_AS", "19_0702_AS", "19_0709_AS"]
if __name__ == "__main__":
    # hci_help("hci")
    main()
