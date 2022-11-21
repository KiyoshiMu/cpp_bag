from collections import defaultdict
from pathlib import Path

import pandas as pd

from cpp_bag.plot import ACCR_LABLE, name_mapping, plot_tag_perf_with_std


METRICS_RENAME_MAP = {
    "precision": "Precision",
    "recall": "Recall",
    "fscore": "F1 Score",
    "Unnamed: 0": "Label",
}
BASE = Path("experiments2")
TRAIL_N = 5
LABEL_N = 5


def merge_metrics(
    prefix="", random_csv=None, write_pdf=False, avg_csv=None, dst_dir=Path("."), hct_csv=None,
):
    metrics = ["Precision", "Recall", "F1 Score"]
    dfs = []

    for trial in range(TRAIL_N):
        DST = BASE / f"trial{trial}"
        ret = pd.read_csv(DST / f"{prefix}{trial}_metric.csv")
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
    for csv, method in zip(csvs ,methods) :
        df = pd.read_csv(csv)
        f1s = [f"{v:.3f}±{s:.3f}" for v, s in zip(df["F1 Score_mean"], df["F1 Score_std"])]
        precisions = [f"{v:.3f}±{s:.3f}" for v, s in zip(df["Precision_mean"], df["Precision_std"])]
        recalls = [f"{v:.3f}±{s:.3f}" for v, s in zip(df["Recall_mean"], df["Recall_std"])]
        methods = [method] * len(f1s)
        labels = [ACCR_LABLE[l] for l in df["Label"]]
        dfs.append(pd.DataFrame({"Label": labels, "F1 Score": f1s, "Precision": precisions, "Recall": recalls, "Method": methods, }))
    df = pd.concat(dfs)
    
    df.to_latex(dst_dir/ "metrics_all.tex", index=False)

if __name__ == "__main__":
    dst_dir = BASE
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
    ret_to_latex(
       csvs , [name_mapping(n.name) for n in csvs], dst_dir=dst_dir
    )
