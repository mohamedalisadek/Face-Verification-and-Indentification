# Face Verification on LFW — Siamese Networks

Mohammed Ali

---

A TensorFlow implementation of Siamese Networks for one-shot face verification, built as an architecture exploration comparing a custom CNN backbone against EfficientNetB0 transfer learning, each under two metric-learning objectives — Contrastive Loss and Triplet Loss — with all loss functions written from scratch.

---

## Overview

This project implements and evaluates Siamese Network architectures for face verification on the [Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) dataset. Four configurations are trained and compared:

| Configuration | Test Accuracy | Test Loss |
|---|---|---|
| Custom CNN + Contrastive Loss | 78.77% | 0.1491 |
| Custom CNN + Triplet Loss | 87.05% | 0.1744 |
| EfficientNetB0 + Contrastive Loss | 80.43% | 0.1355 |
| EfficientNetB0 + Triplet Loss | **88.97%** | **0.1588** |

The best-performing configuration — EfficientNetB0 with Triplet Loss — achieves 88.97% accuracy with the tightest intra-class embedding clustering across all experiments.

---

## Repository Structure

```
.
├── face-verification-lfw-siamese-network.ipynb   # Main notebook
├── saved_models/                                  # Model checkpoints (.keras)
│   ├── best_cnn_contrastive.keras
│   ├── best_cnn_triplet.keras
│   ├── best_efficientnet_contrastive.keras
│   └── best_efficientnet_triplet.keras
└── processed_data/                                # Pair/triplet CSVs (generated at runtime)
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.x (GPU recommended)
- kagglehub
- scikit-learn
- numpy, pandas, matplotlib, tqdm

Install dependencies:

```bash
pip install tensorflow kagglehub scikit-learn numpy pandas matplotlib tqdm
```

The notebook is optimised for Google Colab (GPU runtime) or Kaggle Notebooks. Local CPU execution is possible but significantly slower for training.

---

## Dataset

The LFW dataset is downloaded automatically via KaggleHub at runtime:

```python
lfw_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
```

A two-sided identity filter is applied before any split:

- **Lower bound (MIN_IMAGES = 5):** Removes identities with fewer than 5 images — too sparse for meaningful pair generation.
- **Upper bound (MAX_IMAGES = 50):** Caps heavily represented individuals (e.g. George W. Bush with 530 images) to prevent class imbalance.

After filtering, identities are split at the person level using an open-set 70 / 20 / 10 protocol (train / val / test). No test identity appears during training.

| Split | Identities | Pairs / Triplets per Person | Positive : Negative |
|---|---|---|---|
| Train | ~756 | 40 + 40 | 1:1 |
| Validation | ~216 | 40 + 40 | 1:1 |
| Test | ~108 | 40 + 40 | 1:1 |

---

## How to Run

1. Open `face-verification-lfw-siamese-network.ipynb` in Google Colab or Kaggle (GPU runtime).
2. Run all cells in order from top to bottom. The notebook is fully sequential.
3. The dataset downloads automatically in Section 1.
4. Training runs for up to 40 epochs per model with early stopping.
5. Evaluation and visualisations are produced in Sections 8 and 9.

---

## Notebook Structure

| Section | Description |
|---|---|
| 1 | Imports, global configuration, and hyperparameters |
| 2 | Dataset parsing and identity filtering (EDA included) |
| 3 | Open-set person-level split and pair/triplet generation |
| 4 | tf.data pipeline construction and online augmentation |
| 5 | Custom CNN backbone definition |
| 6 | EfficientNetB0 transfer learning backbone definition |
| 7 | Loss function implementations (Contrastive and Triplet) and model training |
| 8 | Evaluation: test accuracy, distance distribution histograms, ROC curves |
| 9 | End-to-end inference pipeline (`verify_faces` function) |

---

## Model Architectures

### Custom CNN Backbone

A four-block convolutional network built from scratch. Each block applies two consecutive 3x3 convolutions, followed by BatchNormalization, MaxPooling (2x2), and Dropout. Filter depths scale as 32 → 64 → 128 → 256.

- Activation: LeakyReLU (alpha = 0.1) — prevents silent unit collapse observed with standard ReLU
- Pooling: Global Average Pooling — reduces trainable parameters from ~16M to ~800K
- Embedding head: two dense layers projecting to a 128-dimensional L2-normalised space
- Dropout schedule: 0.10 → 0.15 → 0.15 → 0.20 across blocks; 0.20 on the embedding layer

### EfficientNetB0 Backbone (Transfer Learning)

EfficientNetB0 is selected for its compound scaling strategy, yielding strong ImageNet representations (~5.3M parameters). Training uses a two-phase protocol to prevent catastrophic forgetting:

- **Phase 1 — Head warm-up (20 epochs, LR = 3e-4):** The base is frozen; only the 256-D dense head and 128-D embedding layer are trained.
- **Phase 2 — Selective fine-tuning (25 epochs, LR = 1e-5):** The top 50 layers are unfrozen. A very low learning rate allows gentle domain adaptation from ImageNet to LFW.

---

## Loss Functions

Both losses are implemented manually as custom TensorFlow callables. No built-in Keras losses were used.

### Contrastive Loss (margin M = 1.0)

```
L = y * d^2 + (1 - y) * max(M - d, 0)^2
```

Where `d` is the Euclidean distance and `y = 1` for same-person pairs. Accuracy is measured by thresholding at `d = 0.7`.

### Triplet Loss (margin alpha = 0.5)

```
L = max( d(anchor, positive) - d(anchor, negative) + alpha, 0 )
```

True (non-squared) Euclidean distance is used because L2-normalised embeddings live on a unit hypersphere where squared distances distort gradients near the poles. Accuracy is reported as the fraction of triplets satisfying `d(a, p) < d(a, n)` — a threshold-free metric.

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Image size | 160x160 | Reduced from 224 to avoid OOM |
| Batch size | 64 | Increased from 32 for smoother gradient estimates |
| Epochs | 40 | With early stopping (patience = 12) |
| Contrastive margin | 1.0 | |
| Triplet margin | 0.5 | Increased from 0.4 for stronger push on L2-normed space |
| Embedding dimension | 128 | L2-normalised |
| Learning rate (CNN) | 1e-3 | Single phase |
| Learning rate (EffNet Phase 1) | 3e-4 | Head warm-up |
| Learning rate (EffNet Phase 2) | 1e-5 | Fine-tuning |

---

## Augmentation

Online (on-the-fly) augmentation is applied through the `tf.data` pipeline during training only:

- Random crop and resize (0.85–1.0 of original scale)
- Random horizontal flip
- Brightness jitter (±0.15)
- Contrast jitter ([0.85, 1.15])
- Gaussian noise (sigma = 0.008)

Colour jitter is deliberately excluded — skin tone is a valid identity cue.

---

## Inference

The `verify_faces` function accepts two image paths and returns a same/different decision with the Euclidean distance:

```python
result = verify_faces(
    img_path1="path/to/image1.jpg",
    img_path2="path/to/image2.jpg",
    backbone=efficientnet_backbone_t,
    threshold=0.6
)
# Returns: {"distance": 0.3124, "decision": "SAME PERSON"}
```

The recommended backbone for deployment is `efficientnet_backbone_t` (EfficientNetB0 + Triplet Loss).

---

## Results Summary

EfficientNetB0 combined with Triplet Loss achieves the best performance across all metrics. Triplet loss consistently outperforms contrastive loss by 8–10 percentage points, as it enforces relative distance geometry rather than an absolute distance threshold. The two-phase fine-tuning protocol is essential for EfficientNetB0 — skipping Phase 1 caused validation loss to spike due to catastrophic forgetting of the projection head.

Distance distribution plots (Section 8) confirm that the EfficientNetB0 + Triplet configuration produces the tightest intra-class clustering and the widest inter-class margin across all four experiments.

ROC curves and AUC values for all four configurations are generated in the final cells of Section 8.

---

## About

A hands-on exploration of metric learning and Siamese architectures applied to real-world face verification. The project was built to develop practical intuition around backbone design choices, loss function trade-offs, and the effect of transfer learning on embedding quality — with everything implemented from scratch to maximise understanding over convenience.
