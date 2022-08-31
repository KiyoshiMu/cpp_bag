# Whole slide image representation in bone marrow cytology

This a Github Repo hosting custom codes for the paper "*Whole slide image representation in bone marrow cytology*".

## System requirements

### Hardware Requirements

For optimal performance, we recommend a computer with the following specs:

* RAM: 16+ GB
* CPU: 2+ cores, 2.2+ GHz/core
* GPU: 16+ GB

The runtimes below are generated using a computer with the recommended specs:
 * RAM: 16 GB
 * CPU: 2 Intel(R) Xeon(R) CPU @ 2.20GHz
 * GPU: 1 Tesla V100-SXM2-16GB, CUDA Version: 10.1

### Software Requirements

The package development version is tested on Linux operating system (Ubuntu 18.04.5 LTS).

Python Dependencies:

    python = ">=3.9,<3.11"
    faiss-cpu = ">=1.7.2"
    torch = ">=1.10.2"
    numpy = ">=1.18"
    record-keeper = ">=0.9.31"
    tensorboard = ">=2.8.0"
    plotly = ">=5.6.0"
    scikit-learn = ">=1.0.2"
    numba = ">=0.55.1"
    scipy = "^1.8.0"
    umap-learn = ">=0.5.2"
    pytorch-metric-learning = ">=1.2.0"
    tqdm = "^4.63.0"
    kaleido = "^0.2.1"


*It takes about 5-10 mins.*

## Demo

> The data that support the findings of this study are available on reasonable request from the corresponding author, pending local REB and privacy office approval. The data are not publicly available because they contain information that could compromise research participant privacy/consent.

**You need first to contact the corresponding author to get the zipfile data.zip used in for this demo**


## Instructions for use


### Scripts

1. modeling process: [modeling.py](modeling.py)
1. Metrics Figure come from [clf_result.py](clf_result.py)
1. Attention weight plot [plot_att_weight.py](plot_att_weight.py)
1. Embedding Figure [pick_embedding.py](pick_embedding.py)
1. Copy materials for interactive plot [copy_att.py](copy_att.py)
1. Cell type masking [im_score.py](im_score.py)
1. A preview of Cell bag [plot_cell_bag.py](plot_cell_bag.py)