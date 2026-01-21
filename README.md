# Fashion-MNIST Image Classification (MLP Baseline vs CNN)

## Overview
This project trains deep learning models to classify images from the **Fashion-MNIST** dataset (10 clothing categories, 28×28 grayscale).  
I first built a simple **MLP baseline** to validate the data pipeline and training loop, then upgraded to a **CNN** to leverage spatial feature learning.

## Dataset
- **Fashion-MNIST**: 60,000 training images, 10,000 test images  
- Input: 28×28 grayscale image  
- Output: 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

## Models
### 1) MLP Baseline
- Flatten 28×28 → 784
- Linear(784→128) + ReLU
- Linear(128→10)

**Test Accuracy (5 epochs): ~87.14%**

### 2) CNN
- Conv(1→32, 3×3) + ReLU + MaxPool
- Conv(32→64, 3×3) + ReLU + MaxPool
- Flatten → Linear(3136→128) + ReLU + Dropout(0.3) → Linear(128→10)

**Best Test Accuracy: ~91.82%** (saved as `best_cnn_fmnist.pt`)

## Training Setup
- Loss: CrossEntropyLoss  
- Optimizer: Adam (lr=1e-3)  
- Batch size: train=128, test=256  
- Normalization: mean=0.5, std=0.5

## Results
| Model | Test Accuracy |
|------|---------------|
| MLP Baseline | 87.14% |
| CNN | 91.82% |

Upgrading from an MLP to a CNN improved accuracy by **~4.7% absolute**, showing the benefit of spatial feature extraction for image classification.

## Error Analysis
### Confusion Matrix Insights
Most mistakes occur between visually similar upper-body categories, especially:
- **Shirt ↔ T-shirt/top**
- **Shirt ↔ Coat / Pullover**
These classes share similar shapes in low-resolution grayscale images, making them inherently ambiguous.

### High-Confidence Mistakes
I also inspected the **most confident incorrect predictions** (softmax confidence ≈ 0.99–1.00).  
Many errors are between visually similar items (e.g., Sneaker vs Ankle boot), indicating the model relies heavily on coarse shape cues under limited resolution.

## How to Run
1. Install dependencies:
```bash
pip install torch torchvision matplotlib scikit-learn
