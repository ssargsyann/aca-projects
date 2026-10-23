# CIFAR-10 Classification with a Custom CNN

An installable PyTorch project with the original AdvancedCNN architecture, augmented training, validation-selected checkpoints, explicit test evaluation, image inference, a dataset-free demo, and two maintained Jupyter workflows.

[Automated checks](https://github.com/ssargsyann/aca-projects/actions/workflows/cifar10-checks.yml) · [Model card](MODEL_CARD.md) · [Results and verification](docs/results.md) · [Data instructions](data/README.md)

## Result status

**Historical notebook:** saved CUDA output reports **81.56% test accuracy after 15 epochs**, with final printed train loss 0.688. The log is preserved in [historical-log.json](docs/historical-log.json). It monitored test accuracy after every epoch and has not been independently rerun with the original trained weights.

**Maintained implementation:** tested with synthetic data and a short real-CIFAR CPU run. Subset smoke metrics verify the training/evaluation/checkpoint pipeline; they are not a replacement for a full benchmark. See [exact run evidence](docs/results.md). No fresh full 15-epoch reproduction is claimed.

## Structure

```text
configs/default.json          Full 15-epoch training configuration
configs/smoke.json            Short one-epoch real-data verification
src/cifar10_cnn/
  model.py                    Original AdvancedCNN architecture
  data.py                     Disjoint stratified splits and transforms
  engine.py                   Training/evaluation epoch loops
  train.py                    AdamW, cosine scheduling, best validation checkpoint
  checkpoint.py               Strict metadata and weights-only loading
  evaluate.py                 Explicit official-test evaluation
  predict.py                  RGB image inference
  demo.py                     Invented-image workflow
  report.py                   Aggregate training/evaluation figures
  cli.py                      Command-line entry point
tests/                        Network-free regression tests
notebooks/                    Two maintained workflows and historical notebook
data/                         Local dataset instructions
artifacts/                    Ignored data-dependent outputs and checkpoints
docs/                         Result scope and aggregate figures
```

## Install

Python 3.10 or newer. From the collection root:

```bash
cd projects/cifar10-cnn-pytorch
python -m venv .venv
```

Activate `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux. For the tested CPU route:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[reports]"
```

For GPU training, install a compatible torch/torchvision pair using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install this package. `device=auto` uses CUDA when available. Tested Python 3.12 CPU versions are in `constraints-py312-cpu-tested.txt`; +cpu versions come from the official CPU wheel index. This records a tested Windows runtime, not a full cross-platform notebook lock.

## Try without a download

```bash
cifar10-cnn demo --output artifacts/demo
cifar10-cnn report --run artifacts/demo/run --output artifacts/demo/report
```

This trains on invented images/labels, saves and reloads a checkpoint, and classifies a synthetic image. Outputs are marked `SYNTHETIC_DEMO_ONLY`. No benchmark performance is implied. Choose a fresh output directory for each demo or training run.

## Real-data smoke test

```bash
cifar10-cnn download --data-root data
cifar10-cnn train --data-root data --config configs/smoke.json --output artifacts/smoke
cifar10-cnn evaluate --checkpoint artifacts/smoke/best.pt --data-root data --limit 500 --output artifacts/smoke/test-subset.json
cifar10-cnn report --run artifacts/smoke --evaluation artifacts/smoke/test-subset.json --output artifacts/smoke/report
```

The smoke configuration trains one epoch on 1,000 official training images and validates on 200 disjoint images. The separate evaluation command above tests a stratified 500-image subset. These small-run results verify code paths and must not be reported as a fully trained CIFAR-10 benchmark. An existing cache can be used by passing its parent as `--data-root`.

## Full training and final evaluation

```bash
cifar10-cnn train --data-root data --config configs/default.json --output artifacts/full-run
cifar10-cnn evaluate --checkpoint artifacts/full-run/best.pt --data-root data --output artifacts/full-run/test.json
cifar10-cnn report --run artifacts/full-run --evaluation artifacts/full-run/test.json --output artifacts/full-run/report
```

The default trains on 45,000 images and validates on 5,000 images, with 15 epochs, AdamW, and cosine annealing. Augmentation applies only to training rows. The official 10,000-image test set is evaluated only when explicitly requested, and omitting `--limit` evaluates it in full. The current evaluation command runs on CPU. Treat final test evaluation as an assessment after model decisions, not a tuning loop.

Loss is aggregated by sample count, not by averaging unequal batch means. The earliest best-validation checkpoint wins ties. Seeds, selected indices, settings, device, counts, versions, and model size are recorded. Cross-device bit-for-bit determinism is not guaranteed. Checkpoints contain inference weights/metadata, not optimizer state for resumed training.

## Predict one image

```bash
cifar10-cnn predict --checkpoint artifacts/full-run/best.pt --image path/to/image.png --output artifacts/prediction.json
```

Prediction converts to RGB, resizes to 32x32, normalizes exactly as training, and returns all ten softmax scores and the top class. Scores are not calibrated correctness guarantees. Only trusted maintained-format checkpoints are supported; class order, architecture, and normalization are checked. Raw historical state dictionaries are not silently imported.

## Notebooks and tests

```bash
python -m pip install -e ".[notebook]"
jupyter lab
python -m unittest discover -s tests -v
```

Open [quickstart.ipynb](notebooks/quickstart.ipynb) or [real-data-workflow.ipynb](notebooks/real-data-workflow.ipynb), selecting the environment with this package. The real-data notebook uses the smoke configuration by default, not an accidental long training run. Set `CIFAR10_DATA_ROOT` before launching Jupyter to use an existing cache. [Notebook notes](notebooks/README.md) distinguish maintained and original workflows.

CI runs network-free tests, synthetic training/checkpoint inference, and report generation. Dataset files and weights stay local. See the [model card](MODEL_CARD.md), [source data](data/README.md), and [validation record](docs/validation.md).
