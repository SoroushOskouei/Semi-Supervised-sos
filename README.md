# SOS: Semi‑Orchestration‑Supervised  
A new library for a semi‑supervised learning method

**SOS** stands for **S**emi, **O**rchestration, **S**upervised.

## Table of Contents
- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Quick Start](#quick-start)  
- [Usage](#usage)  
- [API Reference](#api-reference)  
- [Contributing](#contributing)  
- [License](#license)  

## Features
- Iterative pseudo‑labeling of unlabeled images  
- Chunk‑based training on unlabeled pools  
- Early stopping based on validation accuracy  
- Transfer learning with popular backbones (DenseNet, ResNet, etc.)  
- Minimal, extensible API  

## Requirements
- Python 3.7+  
- PyTorch 1.7+
- torchvision  
- Pillow  

## Quick Start  
Assuming your data is organized like this:  
    data/  
    ├── labeled/  
    │   ├── class1/  
    │   └── class2/  
    └── unlabeled/  
        ├── img_001.jpg  
        └── img_002.png  

Then:  
    from sos import SemiSupervisedTrainer

    trainer = SemiSupervisedTrainer(
        labeled_dir='data/labeled',
        unlabeled_dir='data/unlabeled',
        k=5,              # number of unlabeled chunks per round
        pseudo_epochs=1,  # train epochs per chunk
        target_acc=0.80,  # early‑stop threshold
    )
    trainer.fit()  

## Usage  
You can customize hyperparameters via the constructor:  
    trainer = SemiSupervisedTrainer(
        labeled_dir='data/labeled',
        unlabeled_dir='data/unlabeled',
        batch_size=128,            # default: 64
        lr=5e-4,                   # default: 1e-4
        max_rounds=20,             # default: 10
        model_fn=torchvision.models.resnet50,  # default: densenet121
        # ... any other args ...
    )
    trainer.fit()  

## API Reference

### `SemiSupervisedTrainer(labeled_dir, unlabeled_dir, **kwargs)`  
Initializes data loaders, model, optimizer, and training parameters.

**Important kwargs**  
- `k` (int): number of unlabeled subsets per round  
- `pseudo_epochs` (int): epochs per pseudo‑labeled chunk  
- `target_acc` (float): early‑stop validation threshold  
- `model_fn` (callable): torchvision model constructor  

### `fit() → float`  
Runs the semi‑supervised training loop (with early stopping). Returns the final test accuracy.

### `evaluate(loader) → float`  
Evaluate the current model on any `DataLoader`. Returns accuracy as a float.

## License  
This project is licensed under the MIT License – see the `LICENSE` file for details.
