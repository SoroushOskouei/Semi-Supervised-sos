# SOSLR: Semi‑Orchestration‑Supervised Learning

A simple and effective **semi‑supervised learning** library for image classification, built with **PyTorch**.

**SOSLR** implements an iterative pseudo‑labeling approach. Starting with a small set of labeled images and a large pool of unlabeled images, it trains a model, predicts *pseudo‑labels* for chunks of the unlabeled data, and then retrains on the combined set. This process repeats, improving the model with each round.

---

## Table of Contents
- [Features](#features)
- [Project & Data Structure](#project--data-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Training](#1-training)
  - [2. Prediction](#2-prediction)
- [API Reference](#api-reference)
  - [`soslr.train(...)`](#soslrtrain)
  - [`soslr.predict(...)`](#soslrpredict)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Simple API** — a clean, function‑based API with `train` and `predict`.
- **Iterative Pseudo‑Labeling** — leverages unlabeled data to improve model accuracy.
- **Transfer Learning** — uses pretrained backbones from `torchvision` (e.g. DenseNet, ResNet).
- **Flexible Stopping Criteria** — stop training by target accuracy / loss *or* by patience‑based early‑stopping.
- **Automatic Artifacts** — saves the best model, class mappings, and transforms for hassle‑free prediction.
- **Reproducibility** — deterministic dataloading with a single `seed` argument.

---

## Project & Data Structure
To use **SOSLR**, we recommend the following folder structure:

```
your-project/
│
├── data/
│   ├── labeled/              # Your initial labeled images
│   │   ├── class_a/
│   │   └── class_b/
│   └── unlabeled/            # Your pool of unlabeled images
│
├── images_to_predict/        # New images for prediction later
│
└── sos_model/                # OUTPUT: where the model will be saved
    ├── best_model.pth
    └── class_mapping.json
```

---

## Installation
```bash
pip install soslr
```

---

## Quick Start

### 1. Training
```python
# run_training.py
import soslr   # pip installed package name

LABELED   = "data/labeled"
UNLABELED = "data/unlabeled"
OUT_DIR   = "sos_model"

final_acc = soslr.train(
    labeled_dir        = LABELED,
    unlabeled_dir      = UNLABELED,
    output_dir         = OUT_DIR,
    model_name         = "resnet50",
    k                  = 5,             # number of unlabeled chunks / round
    pseudo_epochs      = 1,
    stopping_criterion = "target_accuracy",
    target_value       = 0.90,          # e.g. stop at 90 % val‑acc
    patience           = 3
)

print(f"Finished! final test accuracy = {final_acc:.4f}")
```

### 2. Prediction
```python
# run_prediction.py
import soslr
import pprint

PREDICT_DIR = "images_to_predict"
MODEL_DIR   = "sos_model"

predictions = soslr.predict(
    images_dir=PREDICT_DIR,
    model_dir=MODEL_DIR,
    model_name="resnet50"
)

pprint.pprint(predictions)
# ➜ {'new_img1.jpg': 'class_a', 'new_img2.png': 'class_b'}
```

---

## API Reference

### `soslr.train(...)`

Trains a semi‑supervised model and saves the training artifacts.

| Argument            | Type   | Default           | Description |
| ------------------- | ------ | ----------------- | ----------- |
| `labeled_dir`       | `str`  | **Required**      | Path to labeled data (class sub‑folders). |
| `unlabeled_dir`     | `str`  | **Required**      | Path to unlabeled images. |
| `output_dir`        | `str`  | `'sos_model'`     | Where to save the trained model & mapping. |
| `model_name`        | `str`  | `'densenet121'`   | `torchvision` model architecture. |
| `pretrained`        | `bool` | `True`            | Use ImageNet‑pretrained weights. |
| `input_size`        | `int`  | `224`             | Resize images to `input_size × input_size`. |
| `batch_size`        | `int`  | `64`              | Batch size for training & eval. |
| `lr`                | `float`| `1e-4`            | Learning rate (Adam). |
| `k`                 | `int`  | `5`               | Unlabeled chunks per round. |
| `pseudo_epochs`     | `int`  | `1`               | Epochs on each pseudo‑labeled chunk. |
| `max_rounds`        | `int`  | `10`              | Maximum pseudo‑labeling rounds. |
| `val_split`         | `tuple`| `(0.2, 0.2)`      | Fractions of labeled data for val / test. |
| `seed`              | `int`  | `42`              | Random seed for full reproducibility. |
| `stopping_criterion`| `str`  | `'patience_accuracy'` | One of `'target_accuracy'`, `'target_loss'`, `'patience_accuracy'`, `'patience_loss'`. |
| `target_value`      | `float`| `0.98`            | Target metric value used with *target* criteria. |
| `patience`          | `int`  | `3`               | Rounds to wait without improvement for *patience* criteria. |

**Returns:** `float` — final *test* accuracy of the best model.

---

### `soslr.predict(...)`

Makes predictions on a directory of images using a trained model.

| Argument      | Type | Default        | Description |
| ------------- | ---- | -------------- | ----------- |
| `images_dir`  | `str`| **Required**   | Directory containing images to predict. |
| `model_dir`   | `str`| `'sos_model'`  | Directory containing `best_model.pth` & `class_mapping.json`. |
| `model_name`  | `str`| `'densenet121'`| Must match the architecture used during training. |
| `input_size`  | `int`| `224`          | Image size used during training. |

**Returns:** `dict` — mapping `{filename → predicted_class}`.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

---

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
