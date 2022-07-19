from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal
from typing import NamedTuple
from typing import TypedDict

import pandas as pd
from sklearn.metrics import f1_score

from cpp_bag.io_utils import json_load
from cpp_bag.plot import measure_slide_vectors, plot_embedding, plot_tag_perf_with_std


class AdjustFactor(NamedTuple):
    tpr: float
    fnr: float


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
    print(
        f"BERT correct: {bert_correct} ({bert_correct/population:.3f}), Error rate: {1 - bert_correct/population}",
    )
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
    adjust_factor = AdjustFactor(tpr=tpr, fnr=fnr)
    arr_after = adjust_arr(positive, negative, adjust_factor)
    err_after = adjust_negative / population
    print(f"Acc before: {acc_before:.3f}, err before: {err_before:.3f}")
    print(f"Acc after: {arr_after:.3f}, err after: {err_after:.3f}")

    return adjust_factor


def adjust_arr(positive, negative, adjust_factor: AdjustFactor):

    adjust_positive = positive * adjust_factor.tpr + negative * adjust_factor.fnr
    arr_after = adjust_positive / (positive + negative)
    return arr_after


def main(adjust_factor: AdjustFactor = None, export=False):
    accs = []
    w_accs = []
    dummy_accs = []
    w_dummy_accs = []
    for trial in range(5):
        MARK = str(trial)
        DST = Path("data") / MARK
        ret = json_load(DST / f"{MARK}_slide_summary.json")
        if adjust_factor is None:
            accs.append(ret["correct"][1])
        else:
            positive = ret["correct"][0]
            negative = ret["incorrect"][0]
            adjust_acc = adjust_arr(positive, negative, adjust_factor)
            accs.append(adjust_acc)
        w_accs.append(ret["weighted_acc"])

        dummy_ret = json_load(DST / f"{MARK}_slide_dummy_summary.json")
        if adjust_factor is None:
            dummy_accs.append(dummy_ret["correct"][1])
        else:
            positive = dummy_ret["correct"][0]
            negative = dummy_ret["incorrect"][0]
            adjust_acc = adjust_arr(positive, negative, adjust_factor)
            dummy_accs.append(adjust_acc)
        w_dummy_accs.append(dummy_ret["weighted_acc"])
    df = pd.DataFrame(
        dict(acc=accs, w_acc=w_accs, dummy_acc=dummy_accs, w_dummy_acc=w_dummy_accs),
    )
    if export:
        df.to_csv("report.csv", index=False)
    print(df.agg(["mean", "std"]))


METRICS_RENAME_MAP = {
    "precision": "Precision",
    "recall": "Recall",
    "fscore": "F1 Score",
}


def merge_metrics(extra_mark="", random_csv=None, write_pdf=False, avg_csv=None):
    metrics = ["precision", "recall", "fscore"]
    records = defaultdict(list)
    for trial in range(5):
        MARK = str(trial)
        DST = Path("data") / MARK
        ret = pd.read_csv(DST / f"{MARK}{extra_mark}_metric.csv", index_col=0)
        labels = ret.index.to_list()
        for label in labels:
            for metric in metrics:
                records[f"{label}|{metric}"].append(ret.loc[label, metric])
    df = pd.DataFrame(records)
    df.to_csv(f"metrics{extra_mark}.csv", index=False)
    df_agg = df.agg(["mean", "std"]).T
    df_agg.to_csv(f"metrics{extra_mark}_agg.csv")
    metrics_records = defaultdict(list)

    for row in df_agg.iterrows():
        _, _metric = row[0].split("|")
        _mean, _std = row[1]
        metrics_records[f"{METRICS_RENAME_MAP[_metric]}_mean"].append(_mean)
        metrics_records[f"{METRICS_RENAME_MAP[_metric]}_std"].append(_std)
    df_metrics = pd.DataFrame(metrics_records, index=labels)
    df_metrics_dst = f"metrics{extra_mark}_agg_T.csv"
    df_metrics.to_csv(df_metrics_dst)

    main_metrics = "F1 Score"
    include_random = random_csv is not None
    include_avg = avg_csv is not None
    if include_random:
        random_record_df = pd.read_csv(random_csv, index_col=0)
        df_metrics["Dummy_mean"] = random_record_df[f"{main_metrics}_mean"]
        df_metrics["Dummy_std"] = random_record_df[f"{main_metrics}_std"]
    if include_avg:
        print(f"Loading avg csv: {avg_csv}")
        avg_record_df = pd.read_csv(avg_csv, index_col=0)
        df_metrics["Avg_mean"] = avg_record_df[f"{main_metrics}_mean"]
        df_metrics["Avg_std"] = avg_record_df[f"{main_metrics}_std"]
    fig_metrics = plot_tag_perf_with_std(
        df_metrics,
        main_metrics,
        include_random=include_random,
        include_avg=include_avg,
    )
    fig_metrics.write_image(f"metrics{extra_mark}.jpg", scale=2)
    if write_pdf:
        fig_metrics.write_image(f"metrics{extra_mark}.pdf", format="pdf")
    return df_metrics_dst

def draw_embedding(df_p):
    plot_df = pd.read_json(df_p, orient="records")
    fig = plot_embedding(plot_df)
    fig.write_html("embedding.html")
    # fig.write_image("embedding.pdf", format="pdf")

def f1_only():
    for trial in range(5):
        MARK = str(trial)
        DST = Path("data") / MARK
        data = json_load(DST / f"{MARK}.json")
        preds = [item["pred"] for item in data]
        labels = [item["label"] for item in data]
        f1 = f1_score(labels, preds, average="micro")
        print(f"{MARK}: {f1:.3f}")

def avg_pool_f1():
    for trial in range(5):
        MARK = str(trial)
        DST = Path("data") / MARK
        train_pkl = DST / f"train_avg{MARK}_pool.pkl"
        val_pkl = DST / f"val_avg{MARK}_pool.pkl"
        measure_slide_vectors(train_pkl, val_pkl, mark=f"{MARK}_avg", dst=DST, dummy_baseline=False)

if __name__ == "__main__":
    # main(export=False)
    # adjust_factor = review_adjust()
    # main(adjust_factor=adjust_factor, export=True)
    random_csv = merge_metrics(extra_mark="_dummy")
    avg_csv = merge_metrics(extra_mark="_avg")
    merge_metrics(random_csv=random_csv, write_pdf=True, avg_csv=avg_csv)
    # draw_embedding("data/0/0.json")
    # avg_pool_f1()
    # f1_only()