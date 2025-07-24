# Transformers trained on proteins can learn to attend to Euclidean distance

Now [published in TMLR](https://openreview.net/forum?id=mU59bDyqqv)

This directory contains all code used for the paper.

The subdirectories `simulated`, `pretrain`, and `function` correspond to the three experiment sections in the paper.
Model weights can be [found on Zenodo](https://doi.org/10.5281/zenodo.15914050) and the function model predictions are included in this repository which can be used to reproduce the results.

The training scripts for each model are:

* Simulated experiments: `simulated/sim_experiments.py`
* Unconditional pretraining: `pretrain/train.py`
* ESM-conditioned AF-DB pretraining: `pretrain/train_swissprot.py`
* Function prediction: `function/train_[mlp/transformer].py`
