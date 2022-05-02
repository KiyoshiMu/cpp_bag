from __future__ import annotations

from pathlib import Path
from typing import Literal
from typing import TypedDict

import pandas as pd

from cpp_bag.io_utils import json_load


class Synopsis(TypedDict):
    particles: str
    megakaryocytes_and_platelets: str
    extrinsic_cells: str
    erythropoiesis: str
    granulopoiesis: str
    reticulum_cells_plasma_cells_and_lymphocytes: str
    hemosiderin: str
    comment: str


class SlideInfo(TypedDict):
    name: str
    result: Synopsis
    type: str
    patient: int
    tags: str
    tiff_x_resolution: int


class Predict(TypedDict):
    slide_name: str
    bert_label: str
    simple_label: str
    pred: str
    prob_top0: str
    prob_top1: str
    prob_top2: str
    cell_count: int
    pred_entropy: float
    sampled: bool
    top3_correct: bool


Remark = Literal["BERT", "Cell Bag"]


class Review(TypedDict):
    slideName: str
    remark: list[Remark]
    slide_info: SlideInfo
    predict: Predict
    comment: str


def review_adjust():
    """
    Report the error rate of BERT and Cell Bag
    1.
    """
    # def is_bag_correct(review: Review) -> bool:
    #     pred = review["predict"]

    review_p = "data/bag_review.json"
    reviews: list[Review] = json_load(review_p)
    bert_correct_names = set()
    bag_correct_names = set()

    # false_positive_names = set()
    # false_negative_names_ = set()
    positive_names = set()
    negative_names = set()

    for review in reviews:
        if not review["predict"]["top3_correct"] and "Cell Bag" in review["remark"]:
            print("Special cases", review["slideName"])
    # 19_0199_AS
    # 19_0045_AS
    # 19_0585_AS
    # 18_0016_AS
    # 19_0337_AS
    # 19_0319_AS
    # After checking, the review correctly didn't count prob:0.00 case as correct

    # We want to know how much we can trust BERT as the ground truth
    for review in reviews:
        if "BERT" in review["remark"]:
            bert_correct_names.add(review["slideName"])
        if "Cell Bag" in review["remark"]:
            bag_correct_names.add(review["slideName"])

        if review["predict"]["top3_correct"]:
            positive_names.add(review["slideName"])
        else:
            negative_names.add(review["slideName"])

        # if not review["predict"]["top3_correct"] and "Cell Bag" in review["remark"]:
        #     false_negative_names_.add(review["slideName"])
        # if review["predict"]["top3_correct"] and "Cell Bag" not in review["remark"]:
        #     false_positive_names.add(review["slideName"])

    false_positive_names = positive_names - bag_correct_names
    false_negative_names = negative_names & bag_correct_names

    # print(false_negative_names - false_negative_names_)
    bert_correct = len(bert_correct_names)
    bag_correct = len(bag_correct_names)
    positive = len(positive_names)
    negative = len(negative_names)
    false_positive = len(false_positive_names)
    false_negative = len(false_negative_names)
    fpr = false_positive / len(positive_names)
    fnr = false_negative / len(negative_names)
    tpr = 1 - fpr
    tnr = 1 - fnr
    population = len(positive_names) + len(negative_names)
    assert population == len(reviews)
    print(f"Number: {population}")
    print(f"Positive: {positive}")
    print(f"Negative: {negative}")
    print(f"BERT correct: {bert_correct}, Error rate: {1 - bert_correct/population}")
    print(f"Cell Bag correct: {bag_correct}, Error rate: {1 - bag_correct/population}")

    print(f"False positive: {false_positive}")
    print(f"False negative: {false_negative}")
    print(f"False positive rate: {fpr:.3f}, True positive rate: {tpr}")
    print(f"False negative rate: {fnr:.3f}, True negative rate: {tnr}")

    # Overall adjustment
    acc_before = positive / population
    err_before = negative / population

    adjust_positive = positive * tpr + negative * fnr
    adjust_negative = positive * fpr + negative * tnr
    assert adjust_positive + adjust_negative == population
    arr_after = adjust_positive / population
    err_after = adjust_negative / population
    print(f"Acc before: {acc_before:.3f}, err before: {err_before:.3f}")
    print(f"Acc after: {arr_after:.3f}, err after: {err_after:.3f}")
    acc_adjust_factor = arr_after / acc_before
    print(f"Acc adjust factor: {acc_adjust_factor}")

    return acc_adjust_factor


def main(acc_adjust_factor=1.0, export=False):
    accs = []
    w_accs = []
    dummy_accs = []
    w_dummy_accs = []
    for trial in range(5):
        MARK = str(trial)
        DST = Path("data") / MARK
        ret = json_load(DST / f"{MARK}_slide_summary.json")
        accs.append(ret["correct"][1] * acc_adjust_factor)
        w_accs.append(ret["weighted_acc"])

        dummy_ret = json_load(DST / f"{MARK}_slide_dummy_summary.json")
        dummy_accs.append(dummy_ret["correct"][1] * acc_adjust_factor)
        w_dummy_accs.append(dummy_ret["weighted_acc"])
    df = pd.DataFrame(
        dict(acc=accs, w_acc=w_accs, dummy_acc=dummy_accs, w_dummy_acc=w_dummy_accs),
    )
    if export:
        df.to_csv("report.csv", index=False)
    print(df.agg(["mean", "std"]))


if __name__ == "__main__":
    main(export=False)
    acc_adjust_factor = review_adjust()
    main(acc_adjust_factor=acc_adjust_factor, export=True)
