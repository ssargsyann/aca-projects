# CIFAR-10 data

The [official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html) describes 60,000 RGB images of size 32x32 in 10 classes: 50,000 training images and 10,000 test images. It credits Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton and asks users to cite Krizhevsky's *Learning Multiple Layers of Features from Tiny Images* (2009).

Retrieve/verify the Python version with:

```bash
cifar10-cnn download --data-root data
```

The package uses [Torchvision's CIFAR10 loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html), including its source-file integrity checks. The official Python archive MD5 is `c58f30108f718f92721af3b95e74349a`. Only use trusted dataset copies: the official Python batches use pickle serialization. Follow the source's applicable terms; no new dataset license is invented here.

If a local cache already contains `cifar-10-batches-py/`, pass its parent directory with `--data-root`. Training does not download unless `--download` is explicitly supplied. Raw images and archives are kept local and ignored by Git.

## Splits and transforms

The maintained default splits the 50,000 official training images into 45,000 training and 5,000 validation images, stratified by class with seed 42. Training subsets alone receive flip/crop/rotation augmentation. Validation and test images use only tensor conversion and fixed channel normalization (mean/std 0.5). Training requests only `train=True` datasets and does not use official-test images/labels for fitting or selection. Torchvision's integrity checks may still read batch files, including the test file, to verify checksums.

`configs/smoke.json` instead uses balanced subsets of 1,000 training and 200 validation images and one epoch. These verify execution, not benchmark quality. Exact selected training/validation indices are saved in each real-data run. Optional test subset indices are selected deterministically and stratified; full test evaluation is available by omitting `--limit`.

Single-image prediction accepts Pillow-supported images, converts them to RGB, resizes to 32x32, and applies the recorded normalization. Arbitrary photographs can differ substantially from CIFAR-10; softmax scores are not calibrated confidence guarantees.
