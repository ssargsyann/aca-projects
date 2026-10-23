# Model card: custom CIFAR-10 CNN

## Purpose and architecture

Educational classification of 32x32 RGB images into CIFAR-10's ten classes. The original custom model is retained: Conv2d 3→32→64→128→128→256, batch normalization after each convolution, ReLU, three 2x2 pooling stages, elementwise dropout 0.2, and dense 4096→512→10 with dropout 0.5. It is trained from scratch; no pretrained backbone is used. The original code uses ordinary Dropout, not spatial Dropout2d.

## Training and evaluation

The maintained trainer uses augmented training images, a stratified validation partition from the official training data, AdamW, and cosine scheduling. Best validation accuracy selects an inference checkpoint. The test split is evaluated only via a separate command, and no test accuracy is used in checkpoint selection. Seeds and selected indices are recorded, but identical results across hardware/library versions are not guaranteed.

The historical notebook reports 81.56% test accuracy after 15 epochs while monitoring test accuracy every epoch. That saved log is preserved, not presented as a fresh independently reproduced benchmark. The maintained local verification uses synthetic data and a one-epoch real-data subset smoke run. See [the results report](docs/results.md) for exact scope. No fresh full 15-epoch benchmark is claimed.

## Inference and limits

Images are converted to RGB, resized to 32x32, and normalized with mean/std 0.5. Outputs include a class label and ten softmax scores. Scores are not calibrated probabilities of correctness. Domain shift, image quality, and resizing can degrade results. This is not an open-world recognizer, a safety-critical classifier, or a validated production service.

## Data and artifacts

See [data instructions and source](data/README.md). Images and weights remain local. Inference checkpoints use explicit metadata, strict architecture/class order/normalization checks, and restricted weights-only loading. The checkpoint is intended for inference; training resume is outside this implementation. Raw legacy state dictionaries are not accepted automatically by the maintained checkpoint loader.
