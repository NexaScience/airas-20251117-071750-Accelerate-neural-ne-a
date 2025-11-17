"""Data-handling utilities: Dirichlet partitioning, preprocessing and loaders."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


###############################################################################
# Synthetic dataset (for smoke tests)                                          #
###############################################################################


class SyntheticImageDataset(Dataset):
    """Random images & labels. 32×32×3."""

    def __init__(self, n: int = 1024, num_classes: int = 10):
        self.n = n
        self.num_classes = num_classes
        self.x = torch.randn(n, 3, 32, 32)
        self.y = torch.randint(0, num_classes, (n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


###############################################################################
# Dataset registry                                                             #
###############################################################################


DATASET_LOADERS: Dict[str, callable] = {}


def register_dataset(name):
    def _decorator(fn):
        DATASET_LOADERS[name.lower()] = fn
        return fn

    return _decorator


@register_dataset("synthetic")
def _load_synthetic(cfg):
    total = cfg["dataset"].get("total_samples", 1024)
    ds = SyntheticImageDataset(n=total, num_classes=cfg["dataset"].get("num_classes", 10))
    return (ds, ds), 10  # train & test identical here


@register_dataset("cifar10")
def _load_cifar10(cfg):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(28),  # 32 → 28 crop as per spec
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    root = Path(cfg.get("data_dir", "./data"))
    trainset = datasets.CIFAR10(root=str(root), train=True, download=True, transform=train_tf)
    testset = datasets.CIFAR10(root=str(root), train=False, download=True, transform=test_tf)
    return (trainset, testset), 10


###############################################################################
# Preprocessor class                                                           #
###############################################################################


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = cfg["dataset"]["name"].lower()
        if self.dataset_name not in DATASET_LOADERS:
            raise NotImplementedError(
                f"Dataset '{self.dataset_name}' not registered. Add loader via @register_dataset decorator."
            )

    # ------------------------ splitting utilities ------------------------- #

    def _dirichlet_split(self, labels: List[int], num_clients: int, alpha: float):
        labels = np.asarray(labels)
        num_classes = len(np.unique(labels))
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        for c, idxs in enumerate(class_indices):
            np.random.shuffle(idxs)
            proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            split = np.split(idxs, proportions)
            for cid, part in enumerate(split):
                client_indices[cid].extend(part.tolist())
        return client_indices

    # ---------------------------- main API -------------------------------- #

    def get_data_loaders(self) -> Tuple[Dict[int, DataLoader], DataLoader]:
        batch_size = self.cfg["dataset"]["batch_size"]
        num_clients = self.cfg["dataset"]["num_clients"]
        alpha = self.cfg["dataset"].get("alpha", math.inf)

        (trainset, testset), num_classes = DATASET_LOADERS[self.dataset_name](self.cfg)
        self.cfg.setdefault("data", {})
        self.cfg["data"]["num_classes"] = num_classes

        # Partitioning
        if alpha == math.inf:
            indices = np.random.permutation(len(trainset))
            splits = np.array_split(indices, num_clients)
        else:
            labels = [trainset[i][1] for i in range(len(trainset))]
            splits = self._dirichlet_split(labels, num_clients, alpha)

        client_loaders: Dict[int, DataLoader] = {}
        for cid, idxs in enumerate(splits):
            subset = Subset(trainset, idxs)
            client_loaders[cid] = DataLoader(
                subset, batch_size=batch_size, shuffle=True, drop_last=True
            )

        test_loader = DataLoader(testset, batch_size=256, shuffle=False)
        return client_loaders, test_loader
