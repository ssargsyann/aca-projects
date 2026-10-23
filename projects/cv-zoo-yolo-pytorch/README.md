#The CV Zoo & Anatomy of YOLO

A production-grade, self-contained PyTorch/torchvision notebook covering two pillars of modern computer vision: an end-to-end inference pipeline across the four major CV tasks, and a from-scratch, vectorized breakdown of how YOLO assigns grid-cell responsibility and matches anchor boxes via IoU.

---

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
  - [Part 1: The CV Zoo](#part-1-the-cv-zoo)
  - [Part 2: Anatomy of YOLO](#part-2-anatomy-of-yolo)
- [Mathematical Foundations](#mathematical-foundations)
  - [Grid Cell Responsibility](#grid-cell-responsibility)
  - [Intersection over Union (IoU)](#intersection-over-union-iou)
- [Installation](#installation)
  - [Virtual Environment Setup](#virtual-environment-setup)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This project demonstrates a full spectrum of computer vision understanding — from a single image-level label, to bounding boxes, to pixel-perfect instance masks, to dense per-pixel semantic maps — followed by a hands-on deconstruction of the mechanics that make single-shot detectors like YOLO fast and trainable. All models use pretrained ImageNet/COCO weights via the modern `torchvision.models` **Weights Enum API**, and all sample assets are fetched automatically at runtime.

## Core Features

### Part 1: The CV Zoo

| Task | Model | Output |
|---|---|---|
| **Image Classification** | ResNet-50 (`ResNet50_Weights.DEFAULT`) | Top-5 class labels + confidence |
| **Object Detection** | Faster R-CNN ResNet50-FPN v2 | Bounding boxes + labels + scores |
| **Instance Segmentation** | Mask R-CNN ResNet50-FPN v2 | Per-object binary masks + boxes |
| **Semantic Segmentation** | DeepLabV3 (ResNet50 backbone) | Per-pixel class map |

- Automatic device selection (`cuda` → `mps` → `cpu`)
- Modern, non-deprecated `torchvision.models` weights API
- High-resolution visualizations via `torchvision.utils.draw_bounding_boxes` / `draw_segmentation_masks`
- Side-by-side comparison montage across all four tasks

### Part 2: Anatomy of YOLO

- $S \times S$ grid overlay on a real image with ground-truth boxes
- Vectorized computation of **grid-cell responsibility** for every object center
- Custom, fully vectorized **IoU** implementation, numerically validated against `torchvision.ops.box_iou`
- Anchor box template design and IoU-based anchor matching
- Combined visualization of ground-truth box, responsible cell, and winning anchor

## Mathematical Foundations

### Grid Cell Responsibility

YOLO divides an image of size $W_{img} \times H_{img}$ into an $S \times S$ grid. Each cell has dimensions:

$$
w_{cell} = \frac{W_{img}}{S}, \qquad h_{cell} = \frac{H_{img}}{S}
$$

Given an object's ground-truth box center $(x_c, y_c)$, the grid cell **responsible** for detecting that object is indexed by $(i, j)$ (column, row):

$$
i = \left\lfloor \frac{x_c}{w_{cell}} \right\rfloor, \qquad
j = \left\lfloor \frac{y_c}{h_{cell}} \right\rfloor
$$

### Intersection over Union (IoU)

For two axis-aligned boxes $A$ and $B$, IoU quantifies overlap as the ratio of intersection area to union area:

$$
IoU(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}
$$

where $IoU \in [0, 1]$, with $0$ indicating no overlap and $1$ indicating a perfect match. During anchor assignment, the anchor template $a^*$ selected for a ground-truth box $g$ is:

$$
a^* = \underset{a \in \{a_1, \dots, a_B\}}{\arg\max} \; IoU(g, a)
$$

## Installation

### Virtual Environment Setup

**Using `venv`:**

```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

**Using `conda`:**

```bash
conda create -n cv-zoo-yolo python=3.10 -y
conda activate cv-zoo-yolo
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Launch Jupyter and open the notebook:

```bash
jupyter notebook cv_zoo_yolo.ipynb
```

Or run it headlessly end-to-end:

```bash
jupyter nbconvert --to notebook --execute --inplace cv_zoo_yolo.ipynb
```

> **Note:** Part 1 downloads pretrained weights (~100–200 MB total) from `download.pytorch.org` on first run — ensure you have an active internet connection.

## Project Structure

```
\cv-zoo-yolo/
├── cv_zoo_yolo.ipynb      # Main notebook (CV Zoo + Anatomy of YOLO)
├── README.md                  # Project documentation (this file)
├── requirements.txt           # Pinned Python dependencies
├── LICENSE                    # License file
└── assets/                    # (optional) cached sample images
    └── dog.jpg
```

## License

This project is released under the MIT License. See `LICENSE` for details.
