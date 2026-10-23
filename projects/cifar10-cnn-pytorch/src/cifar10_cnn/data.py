"""Deterministic disjoint train/validation partitions and image transforms."""
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def image_transform(training=False):
    steps = []
    if training:
        steps = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.RandomRotation(15)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])


def stratified_limit(indices, targets, limit, seed):
    indices = np.asarray(indices, dtype=int)
    if limit is None:
        return indices
    classes = np.unique(np.asarray(targets)[indices])
    if not isinstance(limit, int) or not len(classes) <= limit <= len(indices):
        raise ValueError("Subset limit must be an integer between class count and available rows.")
    rng = np.random.default_rng(seed)
    count, remainder = divmod(limit, len(classes))
    extra = set(rng.permutation(classes)[:remainder])
    selected = []
    for label in classes:
        available = indices[np.asarray(targets)[indices] == label]
        required = count + int(label in extra)
        if required > len(available):
            raise ValueError("Not enough rows in a class for the requested balanced subset.")
        selected.extend(rng.permutation(available)[:required].tolist())
    return rng.permutation(selected)


def split_indices(targets, fraction=0.1, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    training, validation = [], []
    targets = np.asarray(targets)
    for label in np.unique(targets):
        indices = rng.permutation(np.flatnonzero(targets == label))
        count = round(len(indices) * fraction)
        if not 1 <= count < len(indices):
            raise ValueError("Each class needs rows in both train and validation.")
        validation.extend(indices[:count].tolist())
        training.extend(indices[count:].tolist())
    return rng.permutation(training), rng.permutation(validation)


def training_sets(root, config, download=False):
    augmented = datasets.CIFAR10(root, train=True, download=download, transform=image_transform(True))
    validation_data = datasets.CIFAR10(root, train=True, download=False, transform=image_transform(False))
    train_indices, val_indices = split_indices(augmented.targets, config["validation_fraction"], config["seed"])
    train_indices = stratified_limit(train_indices, augmented.targets, config["train_limit"], config["seed"])
    val_indices = stratified_limit(val_indices, augmented.targets, config["validation_limit"], config["seed"] + 1)
    return Subset(augmented, train_indices.tolist()), Subset(validation_data, val_indices.tolist()), train_indices, val_indices
