import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

class UnlabeledImageFolder(Dataset):
    """Loads images from a folder without requiring subfolders per class."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.paths = [os.path.join(root_dir, fname)
                      for fname in os.listdir(root_dir)
                      if fname.lower().endswith(('.png','.jpg','.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, -1  # dummy label

class SemiSupervisedTrainer:
    def __init__(self,
                 labeled_dir,
                 unlabeled_dir,
                 model_fn=models.densenet121,
                 pretrained=True,
                 input_size=224,
                 batch_size=64,
                 lr=1e-4,
                 k=5,
                 pseudo_epochs=1,
                 max_rounds=10,
                 target_acc=0.80,
                 val_split=(0.2, 0.2),  # val, test proportions
                 seed=42,
                 device=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size, self.lr = batch_size, lr
        self.k, self.pseudo_epochs, self.max_rounds, self.target_acc = k, pseudo_epochs, max_rounds, target_acc

        # transforms
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
        ])

        # prepare datasets & loaders
        self._prepare_data(labeled_dir, unlabeled_dir, val_split)

        # build model
        self.model = model_fn(pretrained=pretrained)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, len(self.labeled_train.dataset.classes))
        self.model = self.model.to(self.device)

        # loss & optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _prepare_data(self, labeled_dir, unlabeled_dir, val_split):
        full = ImageFolder(labeled_dir, transform=self.transform)
        n = len(full)
        idxs = list(range(n))
        random.shuffle(idxs)
        n_val = int(val_split[0] * n)
        n_test = int(val_split[1] * n)
        n_train = n - n_val - n_test

        train_idx = idxs[:n_train]
        val_idx   = idxs[n_train:n_train + n_val]
        test_idx  = idxs[n_train + n_val:]
        self.labeled_train = Subset(full, train_idx)
        self.labeled_val   = Subset(full, val_idx)
        self.labeled_test  = Subset(full, test_idx)
        self.unlabeled_all = UnlabeledImageFolder(unlabeled_dir, transform=self.transform)

        self.train_loader = DataLoader(self.labeled_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_loader   = DataLoader(self.labeled_val, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.test_loader  = DataLoader(self.labeled_test, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def pseudo_label_dataset(self, indices):
        subset = Subset(self.unlabeled_all, indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        pseudo_labels = []
        self.model.eval()
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                outs = self.model(imgs)
                _, preds = outs.max(1)
                pseudo_labels.extend(preds.cpu().tolist())

        class PseudoDataset(Dataset):
            def __init__(self, orig, labels):
                self.orig, self.labels = orig, labels
            def __len__(self):
                return len(self.orig)
            def __getitem__(self, i):
                img, _ = self.orig[i]
                return img, self.labels[i]

        return PseudoDataset(subset, pseudo_labels)

    def train_on(self, dataset, epochs=1):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.model.train()
        for _ in range(epochs):
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(imgs), labels)
                loss.backward()
                self.optimizer.step()

    def fit(self):
        best_val_acc = 0.0
        for rnd in range(1, self.max_rounds+1):
            # split unlabeled into k chunks
            idxs = list(range(len(self.unlabeled_all)))
            random.shuffle(idxs)
            chunks = np.array_split(idxs, self.k)

            print(f"=== Round {rnd}/{self.max_rounds} ===")
            for i, chunk in enumerate(chunks, 1):
                pseudo_ds = self.pseudo_label_dataset(chunk)
                combined = ConcatDataset([self.labeled_train, pseudo_ds])
                self.train_on(combined, epochs=self.pseudo_epochs)
                val_acc = self.evaluate(self.val_loader)
                print(f" Round {rnd}, chunk {i}/{self.k} — Val acc: {val_acc:.4f}")
                best_val_acc = max(best_val_acc, val_acc)

            if best_val_acc >= self.target_acc:
                print(f"Reached target val acc {best_val_acc:.4f}, stopping early.")
                break

        test_acc = self.evaluate(self.test_loader)
        print(f"*** Final TEST accuracy = {test_acc:.4f} ***")
        return test_acc

# Example usage:
# trainer = SemiSupervisedTrainer(
#     labeled_dir='path/to/labeled', unlabeled_dir='path/to/unlabeled'
# )
# trainer.fit()
