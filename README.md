# SFADA

This repository contains source-free domain adaptation experiments for object
detection, including DINO/DETA-based DAOD training, SFOD baselines, sparse
target-label plugins, SageMaker launch utilities, and negative-result paper
aggregation scripts.

## Vendored detrex

`external/detrex` is intentionally vendored into this repository. The DAOD model
adapter resolves detector configs from `external/detrex/projects`, and the
SageMaker Docker image installs detrex from this local source tree. Keeping this
tree in the main repository makes the detector code reproducible after the
original cloud desktop is gone.

The vendored detrex tree started from:

```text
https://github.com/IDEA-Research/detrex.git
commit e244e6c3da3e84566728c52c21fb061d23ce0e2f
```

Local SFADA patch:

```text
external/detrex/projects/deta/modeling/deformable_detr.py
```

The patch clamps DETA inference `topk` to the available number of scores. This
is needed because COCO DETA has many more class scores than DAOD Cityscapes/BDD,
and the original `topk(10000)` can fail after replacing the detector head with
an 8-class DAOD head.

## Detrex Checkpoints

Detector model-zoo checkpoints are not tracked in Git. Download them from the
detrex model zoo:

```text
https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html
```

Place the files under:

```text
external/detrex/ckpts/
```

The adapter looks for relative checkpoint names in that directory. For example,
a config with `detector.init_checkpoint: deta_r50_5scale_12ep_bs8.pth` expects:

```text
external/detrex/ckpts/deta_r50_5scale_12ep_bs8.pth
```

Source-adapted and oracle DAOD checkpoints are experiment artifacts, not detrex
model-zoo checkpoints. Keep those under `runs/daod_source/` and
`runs/daod_oracle/`, or in the corresponding SageMaker/S3 artifact locations.

## Ignored Experiment Artifacts

The following directories are intentionally ignored by Git and should be backed
up separately when needed:

```text
configs/
doc/
runs/
```

For paper/rebuttal use, the most important run artifacts are:

```text
runs/negative_results_summary/
runs/daod_source/
runs/daod_oracle/
runs/baselines/   # metrics/logs are essential; full model weights are optional
```

`external/DDT` and `external/LPLD` are reference checkouts only. Current SFADA
runs use the in-repository implementations under `baselines/`, so those external
reference trees are not required for normal training.
