# ACA Projects

A collection of seven machine learning, computer vision, and natural language processing projects by Sargis Sargsyan.

## Projects

| Project | Focus |
| --- | --- |
| [icu-mortality-prediction](projects/icu-mortality-prediction/README.md) | ICU mortality prediction and model explainability |
| [customer-segmentation-marketing](projects/customer-segmentation-marketing/README.md) | Customer segmentation for marketing |
| [cifar10-cnn-pytorch](projects/cifar10-cnn-pytorch/README.md) | CIFAR-10 image classification with a CNN |
| [cv-zoo-yolo-pytorch](projects/cv-zoo-yolo-pytorch/README.md) | Computer vision experiments with YOLO and PyTorch |
| [unet-yolov8-cv](projects/unet-yolov8-cv/README.md) | Computer vision with U-Net and YOLOv8 |
| [seq2seq-transformer](projects/seq2seq-transformer/README.md) | Sequence-to-sequence Transformer implementation |
| [llm-decoding-strategies](projects/llm-decoding-strategies/README.md) | Language model decoding strategies |

## Getting started

1. Clone this repository:

   ```bash
   git clone https://github.com/ssargsyann/aca-projects.git
   cd aca-projects
   ```

2. Choose a project under `projects/` and read its README.
3. Create a separate Python environment for that project and install its own `requirements.txt`. Dependencies may differ between projects.
4. Open the project's notebook from its project directory. Follow its instructions for datasets, runtime, and any required model downloads.

## Repository layout

Each directory under `projects/` contains a self-contained project. The original Git histories were preserved during consolidation; the import manifest records the imported revisions. Projects can now evolve independently inside this collection.

The [ICU mortality project](projects/icu-mortality-prediction/README.md) now includes a Python package, configurable training, saved-pipeline prediction, optional SHAP plots, and synthetic-data tests. Its original notebook is retained under `notebooks/`. A run on the recovered local dataset achieved validation ROC-AUC 0.8926; see its [run report](projects/icu-mortality-prediction/docs/real-data-results.md) for methodology and limitations. Raw data and model artifacts remain local.

The [customer segmentation project](projects/customer-segmentation-marketing/README.md) also has an installable package, saved-model inference, independent GMM membership diagnostics, synthetic/real Jupyter workflows, and verified aggregate reports. The source cohort's best tested solution was K=2 with in-sample silhouette 0.3605; see its [results](projects/customer-segmentation-marketing/docs/results.md).

The [CIFAR-10 CNN project](projects/cifar10-cnn-pytorch/README.md) now has configurable training, validation-selected checkpoints, separate test evaluation, single-image inference, synthetic/real smoke notebooks, and automated checks. Its original 81.56% notebook log is preserved as historical evidence; the maintained verification is a short CPU smoke run, not a fresh full benchmark. The other four projects retain their original imported layout.

To try the ICU pipeline without a dataset download, install that project and run `icu-mortality demo`, or open its `notebooks/quickstart.ipynb`. The demo uses explicitly synthetic records and arbitrary labels.

The ICU project also provides an executed-and-verified real-data notebook workflow, a model card, tested-version constraints, and reproducible aggregate validation charts. See its README for installation and the provenance note for the remaining data-source limitations.

## Source history

The seven original repository histories are included as ancestors of the import commits. See [the import manifest](docs/import-manifest.json) for source URLs and exact imported revisions. The original repositories are retained.

## Next improvements

- Review dataset access and verify each project's setup and execution steps.
- Extract reusable notebook code into project-specific modules where useful.
- Add focused checks, configuration files, and documented results based on verified runs.
