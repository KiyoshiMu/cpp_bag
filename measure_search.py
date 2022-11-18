"""Measure search Top5 query accuracy"""

import numpy as np
import pandas as pd
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.preprocessing import LabelEncoder

from cpp_bag.io_utils import pkl_load, simplify_label
from cpp_bag.plot import box_plot


def cal_search_quality(
    query,
    reference,
    query_labels,
    reference_labels,
    k=5
):
    accuracy_calculator = AccuracyCalculator(
        include=("mean_average_precision",),
        exclude=(),
        avg_of_avgs=False,
        return_per_class=False,
        k=k,
        label_comparison_fn=None,
        device=None,
        knn_func=None,
        kmeans_func=None,
    )
    ret = accuracy_calculator.get_accuracy(
        query, reference, query_labels, reference_labels, False
    )
    return ret


if __name__ == "__main__":
    from collections import defaultdict
    df_raw = defaultdict(list)
    base = "experiments1"
    for trial in range(5):
        marks = ["Hopfield on Cell Bags",  "HCT", "AvgPooling on Cell Bags",]
        for idx, mark in enumerate(marks) :
            if mark == "Hopfield on Cell Bags":
                train_pkl_p = f"{base}/trial{trial}/train{trial}_pool.pkl"
                val_pkl_p = f"{base}/trial{trial}/val{trial}_pool.pkl"
            elif mark == "AvgPooling on Cell Bags":
                train_pkl_p = f"{base}/trial{trial}/train_avg{trial}_pool.pkl"
                val_pkl_p = f"{base}/trial{trial}/val_avg{trial}_pool.pkl"
            elif mark == "HCT":
                train_pkl_p = f"{base}/trial{trial}/train{trial}_hct.pkl"
                val_pkl_p = f"{base}/trial{trial}/val{trial}_hct.pkl"
            else:
                raise ValueError("Unknown mark")
            train = pkl_load(train_pkl_p)
            val = pkl_load(val_pkl_p)
            train_label = [simplify_label(l) for l in train["labels"]]
            val_label = [simplify_label(l) for l in val["labels"]]
            le = LabelEncoder()
            reference_labels = le.fit_transform(train_label)
            query_labels = le.transform(val_label)
            reference = train["embed_pool"]
            query = val["embed_pool"]
            ret = cal_search_quality(
                query,
                reference,
                query_labels,
                reference_labels,
            )
            print(f"{mark}: {ret}")
            df_raw[mark].append(ret["mean_average_precision"])

            # generate random baseline
            if idx == len(marks) - 1:
                random_query = np.random.rand(*query.shape)
                random_reference = np.random.rand(*reference.shape)
                random_ret = cal_search_quality(
                    random_query,
                    random_reference,
                    query_labels,
                    reference_labels,
                )
                print(f"random: {random_ret}")
                df_raw["Random"].append(random_ret["mean_average_precision"])

    df = pd.DataFrame(df_raw)
    # export to latex with 2 decimal places
    df.to_latex(f"{base}/search_quality.tex", float_format="%.3f")

    # use plotly boxplot to visualize
    
    df = pd.melt(df, var_name="method", value_name="mAP@5")
    fig = box_plot(df, x="method", y="mAP@5")
    # export to pdf
    fig.write_image(f"{base}/search_quality.pdf")

    df.to_csv(f"{base}/search_quality.csv")