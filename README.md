# ACA Projects

Seven machine learning, computer vision and language-model projects by **Sargis Sargsyan**, maintained together with installable Python packages, configurable workflows, executable notebooks, focused tests and measured reports.

## Projects

| Project | Focus | Maintained workflow |
|---|---|---|
| [ICU mortality prediction](projects/icu-mortality-prediction/README.md) | Tabular classification and explainability | Train, saved-pipeline prediction, SHAP and validation reports |
| [Customer segmentation](projects/customer-segmentation-marketing/README.md) | Marketing cohort analysis | Preprocess, PCA, K-Means/GMM, saved-model inference and diagnostics |
| [CIFAR-10 CNN](projects/cifar10-cnn-pytorch/README.md) | Image classification | Configurable training, validation-selected checkpoints, separate test evaluation |
| [CV Zoo and YOLO geometry](projects/cv-zoo-yolo-pytorch/README.md) | Pretrained vision tasks and detection concepts | Four-model inference, IoU/grid/anchor calculations and visual reports |
| [U-Net and YOLOv8](projects/unet-yolov8-cv/README.md) | Segmentation and object detection | Synthetic U-Net training, local COCO8 fine-tuning, image/video inference |
| [Seq2Seq Transformer](projects/seq2seq-transformer/README.md) | Character-level date normalization | Disjoint datasets, encoder-decoder training, greedy evaluation and attention |
| [LLM decoding strategies](projects/llm-decoding-strategies/README.md) | Raw-logit language-model sampling | Temperature/Top-K/Top-P, local GPT-2 generation and reproducible comparisons |

[Verification scope and results](docs/project-status.md) · [GitHub checks](https://github.com/ssargsyann/aca-projects/actions) · [Original source revisions](docs/import-manifest.json)

## Getting started

```bash
git clone https://github.com/ssargsyann/aca-projects.git
cd aca-projects
```

Choose a project above and follow its README. Create a separate Python environment for that project; dependency sets and optional model/data requirements differ. Each guide provides installation, command examples, dataset/model access, notebook instructions and the scope of verified runs.

Several projects have synthetic demos that require no dataset or model download. For example, after installing the relevant project, run `icu-mortality demo`, `cv-zoo demo`, or `llm-decoding demo` with the arguments documented in its README. Real-model commands require the explicitly prepared local assets described there.

## Layout

```text
projects/<project>/
  src/             Project-specific Python package and commands
  configs/         Reproducible workflow settings
  tests/           Focused regression checks
  notebooks/       Maintained workflows and preserved original notebook
  docs/            Measured results, figures, provenance and validation
  README.md        Setup and execution guide
```

Project-specific data, assets and output directories are documented where applicable. Large datasets, weights and local run outputs are ignored; committed reports contain small figures and auditable aggregate/prediction records. Source access, hashes and licensing notes are recorded per project rather than assumed to be shared by the whole collection.

## Evidence and reproducibility

Each project has an automated workflow and documented local validation. Lightweight checks are separate from real-data/model experiments. Reports distinguish fresh measured runs, historical notebook logs, synthetic demonstrations and small execution checks. They also disclose limitations such as incomplete source provenance, tiny datasets, pretrained overlap, qualitative predictions and unverified full-training presets.

Read [the project verification table](docs/project-status.md) before interpreting scores. These are educational/research implementations; a successful run is not a generalization, production-readiness or deployment claim.

## Source history

The original histories of all seven repositories are preserved as ancestors of the consolidation commits. The [import manifest](docs/import-manifest.json) records their URLs and exact imported revisions. Original notebooks are retained under each project's `notebooks/`, and the original repositories remain available.
