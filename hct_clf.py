from cpp_bag import data

import numpy as np

from cpp_bag.io_utils import json_load, pkl_dump

from cpp_bag.performance import create_knn
from cpp_bag.performance import dump_metric


from pathlib import Path

Lineage = """Neutrophil,Metamyelocyte,Myelocyte,\
Promyelocyte,Blast,Erythroblast,Megakaryocyte_nucleus,\
Lymphocyte,Monocyte,Plasma_cell,Eosinophil,Basophil,\
Histiocyte,Mast_cell""".split(
    ",",
)


def main():
    all_cells = data.load_cells()

    dataset = data.CustomImageDataset(
        data.FEAT_DIR,
        data.LABEL_DIR,
        bag_size=256,
        cell_threshold=256,
        with_MK=False,
        all_cells=all_cells,
    )

    slide_portion = dataset.slide_portion

    labels = dataset.le.inverse_transform(dataset.targets)

    x_vec = count_2_vec(slide_portion, Lineage)

    BASE_DIR = "experiments2"
    base = Path(BASE_DIR)
    n = 5
    mark = "hct"
    for trial in range(n):
        dst_dir = base / f"trial{trial}"
        split_json_p = dst_dir / f"split{trial}.json"
        split = json_load(split_json_p)
        train_index, test_index = split["train"], split["val"]
        x_train = x_vec[train_index]
        x_test = x_vec[test_index]

        y_train = labels[train_index]
        y_test = labels[test_index]

        knn = create_knn(x_train, y_train)
        pkl_dump(dict(embed_pool=x_train, labels=y_train, index=train_index), dst_dir / f"train{trial}_{mark}.pkl") 
        pkl_dump(dict(embed_pool=x_test, labels=y_test, index=test_index), dst_dir / f"val{trial}_{mark}.pkl")
        classes_ = knn.classes_
        preds = knn.predict(x_test)
        dump_metric(y_test, preds, classes_, dst_dir / f"{mark}{trial}_metric.csv")


def count_2_vec(slide_portion, Lineage):
    vec = np.zeros((len(slide_portion), len(Lineage)))
    for i, portion in enumerate(slide_portion):
        vec[i] += [portion.get(c, 0) for c in Lineage]
        vec[i] /= sum(portion.values())
    return vec


if __name__ == "__main__":
    main()