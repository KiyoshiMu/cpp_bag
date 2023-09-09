from collections import defaultdict
from pathlib import Path

import pandas as pd

from cpp_bag.plot import ACCR_LABLE, name_mapping, plot_tag_perf_with_std
from scipy import stats

METRICS_RENAME_MAP = {
    "precision": "Precision",
    # "recall": "Recall",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "fscore": "F1 Score",
    "Unnamed: 0": "Label",
}
# BASE = Path("experiments2")
TRAIL_N = 5
LABEL_N = 5


def merge_metrics(
    prefix="",
    random_csv=None,
    write_pdf=False,
    avg_csv=None,
    dst_dir=Path("."),
    hct_csv=None,
):
    metrics = ["Precision", "Sensitivity", "F1 Score", "Specificity"]
    dfs = []

    for trial in range(TRAIL_N):
        ret = pd.read_csv(dst_dir / f"{prefix}{trial}_metric.csv")
        ret.rename(columns=METRICS_RENAME_MAP, inplace=True)
        ret["Trial"] = trial
        ret_melt = ret.melt(id_vars=["Label", "Trial"], value_vars=metrics)
        dfs.append(ret_melt)

    df = pd.concat(dfs)
    assert len(df) == TRAIL_N * LABEL_N * len(
        metrics
    ), "aggregated dataframe is not correct"
    df.to_csv(dst_dir / f"metrics{prefix}.csv", index=False)
    df_agg = df.groupby(["Label", "variable"])["value"].agg(["mean", "std"])
    df_agg.to_csv(dst_dir / f"metrics{prefix}_agg.csv")

    metrics_records = defaultdict(list)
    labels = df_agg.index.get_level_values("Label").unique()
    for row in df_agg.iterrows():
        (
            _,
            _metric,
        ) = row[0]
        _mean, _std = row[1]
        metrics_records[f"{_metric}_mean"].append(_mean)
        metrics_records[f"{_metric}_std"].append(_std)
    df_metrics = pd.DataFrame(metrics_records, index=labels)
    df_metrics_dst = dst_dir / f"metrics{prefix}_agg_T.csv"
    df_metrics.to_csv(df_metrics_dst)

    main_metrics = "F1 Score"
    include_random = random_csv is not None
    include_avg = avg_csv is not None
    include_hct = hct_csv is not None
    if include_random:
        random_record_df = pd.read_csv(random_csv, index_col=0)
        df_metrics["Dummy_mean"] = random_record_df[f"{main_metrics}_mean"]
        df_metrics["Dummy_std"] = random_record_df[f"{main_metrics}_std"]
    if include_avg:
        avg_record_df = pd.read_csv(avg_csv, index_col=0)
        df_metrics["Avg_mean"] = avg_record_df[f"{main_metrics}_mean"]
        df_metrics["Avg_std"] = avg_record_df[f"{main_metrics}_std"]
    if include_hct:
        hct_record_df = pd.read_csv(hct_csv, index_col=0)
        df_metrics["HCT_mean"] = hct_record_df[f"{main_metrics}_mean"]
        df_metrics["HCT_std"] = hct_record_df[f"{main_metrics}_std"]
    fig_metrics = plot_tag_perf_with_std(
        df_metrics,
        main_metrics,
        include_random=include_random,
        include_avg=include_avg,
        include_hct=include_hct,
        show_recall_precision=False,
    )
    fig_metrics.write_image(str(dst_dir / f"metrics{prefix}.jpg"), scale=2)
    if write_pdf:
        fig_metrics.write_image(str(dst_dir / f"metrics{prefix}.pdf"), format="pdf")
    return df_metrics_dst


def ret_to_latex(csvs, methods, dst_dir=Path(".")):
    dfs = []
    for csv, method in zip(csvs, methods):
        df = pd.read_csv(csv)
        f1s = [
            f"{v:.3f}±{s:.3f}" for v, s in zip(df["F1 Score_mean"], df["F1 Score_std"])
        ]
        precisions = [
            f"{v:.3f}±{s:.3f}"
            for v, s in zip(df["Precision_mean"], df["Precision_std"])
        ]
        recalls = [
            f"{v:.3f}±{s:.3f}"
            for v, s in zip(df["Sensitivity_mean"], df["Sensitivity_std"])
        ]
        specialities = [
            f"{v:.3f}±{s:.3f}"
            for v, s in zip(df["Specificity_mean"], df["Specificity_std"])
        ]
        methods = [method] * len(f1s)
        labels = [ACCR_LABLE[l] for l in df["Label"]]
        dfs.append(
            pd.DataFrame(
                {
                    "Label": labels,
                    "F1 Score": f1s,
                    "Precision": precisions,
                    "Sensitivity": recalls,
                    "Specificity": specialities,
                    "Method": methods,
                }
            )
        )
    df = pd.concat(dfs)

    df.to_latex(dst_dir / "metrics_all.tex", index=False)


def average_metrics(csv_p):
    # calculate the weighted average of precision, recall, specificity, fscore
    df = pd.read_csv(csv_p)
    df["support"] = df["tp"] + df["fn"]
    df["weight"] = df["support"] / df["support"].sum()
    precision = (df["precision"] * df["weight"]).sum()
    sensitivity = (df["sensitivity"] * df["weight"]).sum()
    specificity = (df["specificity"] * df["weight"]).sum()
    fscore = (df["fscore"] * df["weight"]).sum()
    return dict(
        precision=precision,
        sensitivity=sensitivity,
        specificity=specificity,
        fscore=fscore,
    )


def cv_agg(csvs, dst_dir, marker="pool"):
    metrics = [average_metrics(csv_p) for csv_p in csvs]
    df = pd.DataFrame(metrics)
    df.to_csv(dst_dir / f"{marker}_cv_agg.csv")
    df_agg = df.aggregate(["mean", "std"])
    precision = f"{df_agg['precision']['mean']:.3f}±{df_agg['precision']['std']:.3f}"
    sensitivity = (
        f"{df_agg['sensitivity']['mean']:.3f}±{df_agg['sensitivity']['std']:.3f}"
    )
    specificity = (
        f"{df_agg['specificity']['mean']:.3f}±{df_agg['specificity']['std']:.3f}"
    )
    fscore = f"{df_agg['fscore']['mean']:.3f}±{df_agg['fscore']['std']:.3f}"
    return dict(
        precision=precision,
        sensitivity=sensitivity,
        specificity=specificity,
        fscore=fscore,
    )


def cv_agg_search(csv):
    df = pd.read_csv(csv)
    df_agg = df.groupby("method")["mAP@10"].agg(["mean", "std"])
    pool = df_agg.loc["Hopfield on Cell Bags"]
    hct = df_agg.loc["rHCT"]
    avg = df_agg.loc["AvgPooling on Cell Bags"]
    dummy = df_agg.loc["Random"]
    aggs = [
        f"{pool['mean']:.3f}±{pool['std']:.3f}",
        f"{hct['mean']:.3f}±{hct['std']:.3f}",
        f"{avg['mean']:.3f}±{avg['std']:.3f}",
        f"{dummy['mean']:.3f}±{dummy['std']:.3f}",
    ]
    print(aggs)
    g1 = df.loc[df["method"] == "Hopfield on Cell Bags", "mAP@10"]
    g2 = df.loc[df["method"] == "rHCT", "mAP@10"]
    print(stats.ttest_rel(g1, g2, alternative="greater"))

    
def cv_ttest(csv1, csv2):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    for metric in df1.columns:
        if metric.startswith("Unnamed"):
            continue
        print(metric)
        print(stats.ttest_rel(df1[metric], df2[metric], alternative="greater"))


if __name__ == "__main__":
    dst_dir = Path("clf_ret")
    random_csv = merge_metrics(prefix="dummy", dst_dir=dst_dir)
    avg_csv = merge_metrics(prefix="avg", dst_dir=dst_dir)
    hct_csv = merge_metrics(prefix="hct", dst_dir=dst_dir)
    merge_metrics(
        prefix="pool",
        random_csv=random_csv,
        write_pdf=True,
        avg_csv=avg_csv,
        dst_dir=dst_dir,
        hct_csv=hct_csv,
    )
    csvs = sorted(dst_dir.glob("metrics*_T.csv"))
    ret_to_latex(csvs, [name_mapping(n.name) for n in csvs], dst_dir=dst_dir)

    cv_agg_search("experiments2/search_quality.csv")
    pool_agg = cv_agg(
        list(Path("clf_ret").glob("pool*_metric.csv")),
        dst_dir=Path("clf_ret"),
        marker="pool",
    )
    hct_agg = cv_agg(
        list(Path("clf_ret").glob("hct*_metric.csv")),
        dst_dir=Path("clf_ret"),
        marker="hct",
    )
    avg_agg = cv_agg(
        list(Path("clf_ret").glob("avg*_metric.csv")),
        dst_dir=Path("clf_ret"),
        marker="avg",
    )
    dummy_agg = cv_agg(
        list(Path("clf_ret").glob("dummy*_metric.csv")),
        dst_dir=Path("clf_ret"),
        marker="dummy",
    )
    df = pd.DataFrame([pool_agg, hct_agg, avg_agg, dummy_agg])
    df.index = ["Hopfield on Cell Bags", "rHCT", "AvgPooling on Cell Bags", "Guessing"]
    df.columns = ["Precision", "Sensitivity", "Specificity", "F1 Score"]
    df.to_csv(Path("clf_ret") / "cv_agg.csv")
    df.to_latex(Path("clf_ret") / "cv_agg.tex")
    cv_ttest("clf_ret/pool_cv_agg.csv", "clf_ret/hct_cv_agg.csv")
