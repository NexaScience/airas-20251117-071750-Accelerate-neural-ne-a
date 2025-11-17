"""Data-handling utilities.

Supports:
• Synthetic random images (for smoke tests)
• CIFAR-10 / CIFAR-100 with torchvision

Provides `Preprocessor` that performs Dirichlet client partitioning and returns
    client_id -> DataLoader  (train)
    + global test DataLoader
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


################################################################################
# Synthetic dataset (tiny, fast)                                               #
################################################################################


class SyntheticImageDataset(Dataset):
    """Random images & labels for smoke testing.  32×32×3."""

    def __init__(self, n: int = 1024, num_classes: int = 10):
        self.n = n
        self.num_classes = num_classes
        self.x = torch.randn(n, 3, 32, 32)
        self.y = torch.randint(0, num_classes, (n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


################################################################################
# Dataset loader registry                                                      #
################################################################################

DATASET_LOADERS: Dict[str, callable] = {}


def register_dataset(name):
    def decorator(fn):
        DATASET_LOADERS[name.lower()] = fn
        return fn

    return decorator


@register_dataset("synthetic")
def _load_synthetic(cfg):
    total = cfg["data"].get("total_samples", 1024)
    num_classes = cfg["data"].get("num_classes", 10)
    ds = SyntheticImageDataset(n=total, num_classes=num_classes)
    return ds, num_classes


################################################################################
# CIFAR-10 / CIFAR-100 loaders                                                 #
################################################################################

_COMMON_CIFAR_MEAN = {
    10: (0.4914, 0.4822, 0.4465),
    100: (0.5071, 0.4867, 0.4408),
}
_COMMON_CIFAR_STD = {
    10: (0.2023, 0.1994, 0.2010),
    100: (0.2675, 0.2565, 0.2761),
}


def _cifar_transforms(num_classes: int):
    mean = _COMMON_CIFAR_MEAN[num_classes]
    std = _COMMON_CIFAR_STD[num_classes]
    train_tfm = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),  # 32 -> 28 as spec
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return train_tfm, test_tfm


@register_dataset("cifar10")
def _load_cifar10(cfg):
    root = Path(cfg.get("data_dir", "./data"))
    train_tfm, test_tfm = _cifar_transforms(10)
    trainset = datasets.CIFAR10(root=str(root), train=True, transform=train_tfm, download=True)
    testset = datasets.CIFAR10(root=str(root), train=False, transform=test_tfm, download=True)
    return (trainset, testset), 10


@register_dataset("cifar100")
def _load_cifar100(cfg):
    root = Path(cfg.get("data_dir", "./data"))
    train_tfm, test_tfm = _cifar_transforms(100)
    trainset = datasets.CIFAR100(root=str(root), train=True, transform=train_tfm, download=True)
    testset = datasets.CIFAR100(root=str(root), train=False, transform=test_tfm, download=True)
    return (trainset, testset), 100


################################################################################
# Preprocessor                                                                 #
################################################################################


class Preprocessor:
    """Prepare per-client DataLoaders & global test loader."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = cfg["dataset"]["name"].lower()
        if self.dataset_name not in DATASET_LOADERS:
            raise NotImplementedError(
                f"Dataset '{self.dataset_name}' not registered. Add a loader via @register_dataset decorator."
            )

    # ---------------- Dirichlet Split ---------------- #
    def _dirichlet_split(self, labels: List[int], num_clients: int, alpha: float):
        labels = np.array(labels)
        num_classes = len(np.unique(labels))
        # Pre-compute indices per class
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        for c, idxs in enumerate(class_indices):
            np.random.shuffle(idxs)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Balance to ensure each client gets at least one sample per class
            proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            splits = np.split(idxs, proportions)
            for cid, part in enumerate(splits):
                client_indices[cid].extend(part.tolist())
        return client_indices

    # ---------------- Public API ---------------- #
    def get_data_loaders(self) -> Tuple[Dict[int, DataLoader], DataLoader]:
        batch_size = self.cfg["dataset"]["batch_size"]
        num_clients = self.cfg["dataset"]["num_clients"]
        alpha = self.cfg["dataset"].get("alpha", math.inf)

        dataset_obj, num_classes = DATASET_LOADERS[self.dataset_name](self.cfg)
        # Store for downstream use
        self.cfg.setdefault("data", {})
        self.cfg["data"]["num_classes"] = num_classes

        # Handle (train, test) tuple or single dataset
        if isinstance(dataset_obj, tuple):
            trainset, testset = dataset_obj
        else:
            trainset = dataset_obj
            testset = dataset_obj

        # Create client splits
        if alpha == math.inf:
            indices = np.arange(len(trainset))
            np.random.shuffle(indices)
            splits = np.array_split(indices, num_clients)
        else:
            labels = [trainset[i][1] for i in range(len(trainset))]
            splits = self._dirichlet_split(labels, num_clients, alpha)

        client_loaders: Dict[int, DataLoader] = {}
        for cid, idx in enumerate(splits):
            subset = Subset(trainset, idx)
            client_loaders[cid] = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=len(subset) >= batch_size,
                num_workers=2,
            )

        test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
        return client_loaders, test_loader
