from __future__ import annotations

import csv
import math

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

from cpp_bag.io_utils import pkl_load
from cpp_bag.io_utils import simplify_label


def load_size(fp="data/slide_size.csv"):
    with open(fp, "r") as f:
        reader = csv.reader(f)
        next(reader)
        return {row[0]: int(row[1]) for row in reader}


def create_knn(refer_embed: np.ndarray, labels):
    n_neighbors = round(math.sqrt(len(refer_embed)))
    print(f"n_neighbors: {n_neighbors}")
    knn: KNeighborsClassifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
    ).fit(
        refer_embed,
        labels,
    )
    return knn


def performance_measure(train_pkl_p, val_pkl_p, mark="pool", random_base=False):
    train = pkl_load(train_pkl_p)
    test = pkl_load(val_pkl_p)
    labels = [simplify_label(l) for l in train["labels"]]
    unique_labels = sorted(set(labels))
    refer_embed = train["embed_pool"]
    knn = create_knn(refer_embed, labels)
    print(refer_embed.shape)
    y_pred = knn.predict(test["embed_pool"])
    y_true = [simplify_label(l) for l in test["labels"]]
    dump_metric(y_true, y_pred, unique_labels, mark=f"data/{mark}_metric.csv")

    if random_base:
        dummy = DummyClassifier(strategy="stratified", random_state=42).fit(
            refer_embed,
            labels,
        )
        y_pred = dummy.predict(test["embed_pool"])
        dump_metric(y_true, y_pred, unique_labels, mark="data/dummy_metric.csv")


def dump_metric(y_true, y_pred, unique_labels, dst, to_csv=True):
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=unique_labels,
    )
    # print(precision, recall, fscore)
    if to_csv:
        metric_df = pd.DataFrame(
            dict(precision=precision, recall=recall, fscore=fscore),
            index=unique_labels,
        )

        metric_df.to_csv(dst)


def cal_weighted_acc(label, *preds):
    acc = 0
    for rank, pred in enumerate(preds, start=1):
        confident = float(pred.split(":")[-1])
        acc += int(label in pred and "0.00" not in pred) * confident
    return acc


def proba_to_dfDict(pred_probs, classes_, val_labels):

    pred_probs_argsort = np.argsort(pred_probs, axis=1)[:, ::-1]
    prob_top0 = [
        f"{classes_[indices[0]]}:{pred_probs[row_idx, indices[0]]:.2f}"
        for row_idx, indices in enumerate(pred_probs_argsort)
    ]
    prob_top1 = [
        f"{classes_[indices[1]]}:{pred_probs[row_idx, indices[1]]:.2f}"
        for row_idx, indices in enumerate(pred_probs_argsort)
    ]
    prob_top2 = [
        f"{classes_[indices[2]]}:{pred_probs[row_idx, indices[2]]:.2f}"
        for row_idx, indices in enumerate(pred_probs_argsort)
    ]
    top3_corrects = [
        any(
            e
            for e in (prob_top0[idx], prob_top1[idx], prob_top2[idx])
            if ("0.00" not in e and val_labels[idx] in e)
        )
        for idx in range(len(val_labels))
    ]
    weighted_acc = [
        cal_weighted_acc(
            val_labels[idx],
            prob_top0[idx],
            prob_top1[idx],
            prob_top2[idx],
        )
        for idx in range(len(val_labels))
    ]
    _df = {
        "label": val_labels,
        "prob_top0": prob_top0,
        "prob_top1": prob_top1,
        "prob_top2": prob_top2,
        "top3_correct": top3_corrects,
        "weighted_acc": weighted_acc,
    }
    return _df


def top3_summary(cases):
    correct_cases = cases[cases["top3_correct"]]
    incorrect_cases = cases[~cases["top3_correct"]]
    weighted_acc_mean = cases["weighted_acc"].mean()
    summary = {
        "correct": (len(correct_cases), len(correct_cases) / len(cases)),
        "incorrect": (len(incorrect_cases), len(incorrect_cases) / len(cases)),
        "weighted_acc": weighted_acc_mean,
    }
    return summary


def dummy_exp(refer_embed, refer_labels, test_embed, test_labels, dst):
    dummy = DummyClassifier(strategy="stratified", random_state=42).fit(
        refer_embed,
        refer_labels,
    )
    classes_ = dummy.classes_
    pred_probs = dummy.predict_proba(test_embed)
    pred = dummy.predict(test_embed)
    dump_metric(test_labels, pred, classes_, dst)
    _df = proba_to_dfDict(pred_probs, classes_, test_labels)
    summary = top3_summary(pd.DataFrame(_df))
    return summary


if __name__ == "__main__":
    performance_measure(
        "data/train_embed_pool.pkl",
        "data/val_embed_pool.pkl",
    )
