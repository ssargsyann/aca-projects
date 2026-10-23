# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)

This repository contains a PyTorch implementation of a custom Convolutional Neural Network (CNN) designed to classify images from the classic CIFAR-10 dataset into 10 distinct categories.

The primary focus of this project is to build an optimized, non-trivial network architecture from scratch (without utilizing pre-trained transfer learning models) and achieve high test accuracy using proper regularization and scheduling techniques.

---

## 🚀 Model Performance Summary

The model was trained for **15 epochs** using the standard training and test splits. Thanks to the integrated cosine annealing learning rate scheduling, the network successfully converged to a highly stable state with a sharp decrease in loss and steady accuracy growth.

*   **Peak Test Accuracy:** `81.56%`
*   **Final Training Loss:** `0.688`
*   **Hardware Accelerator:** NVIDIA T4 GPU (Google Colab Compute)

---

## 🛠️ Key Architectural Features

Instead of a basic sequential stack, the model incorporates modern regularization techniques to prevent overfitting and guarantee smooth training updates:
*   **Batch Normalization:** Applied after every convolutional block to stabilize internal covariate shift and accelerate convergence.
*   **Dropout Regularization:** Strategically placed spatial dropout layers to prevent co-adaptation of features.
*   **Advanced Optimization:** Trained using the AdamW optimizer coupled with a `CosineAnnealingLR` learning rate scheduler to smoothly decay learning rates for fine-tuning in the later stages.

---

## 🧠 Custom Architecture vs. Transfer Learning (Discussion)

A key constraint and choice in this project was building the network **entirely from scratch** (custom weight initialization without relying on pre-trained backbones like ResNet or VGG). 

*   **Why Scratch?** Training a custom CNN from scratch on CIFAR-10 serves as a solid baseline to prove the effectiveness of basic convergence strategies (Batch Normalization, Dropout, Cosine Decay). Reaching **~81.56%** accuracy with a shallow, lightweight model demonstrates that the architecture is structurally sound and regularized correctly.
*   **The Transfer Learning Alternative:** If this project were scaled using Transfer Learning—for instance, fine-tuning a pre-trained **ResNet-50** or **EfficientNet** backbone (pre-trained on ImageNet)—the test accuracy would easily scale to **90%+**. However, using a massive pre-trained architecture bypasses the educational and structural challenge of training a stable optimization loop from the ground up.

---

## 📊 Detailed Training History

Below is the structured output log reflecting the final convergence behavior of the network:

| Epoch | Training Loss | Test Accuracy |
| :---: | :-----------: | :-----------: |
| 1     | 1.704         | 49.19%        |
| 2     | 1.351         | 58.41%        |
| 3     | 1.205         | 65.84%        |
| 4     | 1.117         | 66.99%        |
| 5     | 1.035         | 69.51%        |
| 6     | 0.967         | 73.14%        |
| 7     | 0.909         | 75.60%        |
| 8     | 0.863         | 74.75%        |
| 9     | 0.819         | 77.86%        |
| 10    | 0.785         | 78.92%        |
| 11    | 0.758         | 80.51%        |
| 12    | 0.729         | 80.68%        |
| 13    | 0.705         | 81.06%        |
| 14    | 0.698         | 81.38%        |
| **15**| **0.688**     | **81.56%**    |


## ⚙️ How to Run this Project

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/ssargsyann/cifar10-cnn-pytorch.git](https://github.com/ssargsyann/cifar10-cnn-pytorch.git)
   cd cifar10-cnn-pytorch