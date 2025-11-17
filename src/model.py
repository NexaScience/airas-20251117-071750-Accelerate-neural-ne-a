"""Model architectures and FL-specific wrappers (FedAvg, FedMPQ, FedMPQ-KD, FedPQ)."""
from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Knowledge-distillation helper                                               #
###############################################################################

def kd_loss(logits_student: torch.Tensor, logits_teacher: torch.Tensor, T: float = 2.0):
    p_s = F.log_softmax(logits_student / T, dim=1)
    p_t = F.softmax(logits_teacher.detach() / T, dim=1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)


###############################################################################
# Quantisation utilities                                                      #
###############################################################################

def quantize_tensor(tensor: torch.Tensor, num_bits: int = 8):
    """Symmetric uniform quantisation to signed integers."""
    if num_bits >= 32:
        return tensor  # effectively no quantisation
    qmax = 2 ** (num_bits - 1) - 1
    max_val = tensor.abs().max() + 1e-8  # avoid division by zero
    scale = max_val / qmax
    qt = torch.round(tensor / scale).clamp(-qmax, qmax)
    return qt * scale


###############################################################################
# Base models                                                                 #
###############################################################################

# --- Tiny CNN (used in smoke tests) ---------------------------------------- #


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --- ResNet-20 for CIFAR-10 ------------------------------------------------- #


def _conv3x3(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet_CIFAR(nn.Module):
    def __init__(self, num_blocks: List[int], num_classes: int = 10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = _conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * _BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * _BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet20(num_classes=10):
    return ResNet_CIFAR([3, 3, 3], num_classes=num_classes)


# --------------------------------------------------------------------------- #

BASE_MODEL_FACTORY = {
    "tiny_cnn": TinyCNN,
    "resnet20": resnet20,
}


def get_base_model(name: str, num_classes: int):
    name = name.lower()
    if name not in BASE_MODEL_FACTORY:
        raise NotImplementedError(f"Unknown architecture '{name}'. Available: {list(BASE_MODEL_FACTORY)}")
    return BASE_MODEL_FACTORY[name](num_classes=num_classes)


###############################################################################
# Wrappers                                                                     #
###############################################################################


class FedMPQWrapper(nn.Module):
    """Mixed-precision quantisation + bit-regularisation (original FedMPQ)."""

    def __init__(self, base_model: nn.Module, cfg: Dict):
        super().__init__()
        self.base_model = base_model
        self.bits = cfg["model"].get("bits", 8)

    # -------------------- FL specific helpers ----------------------------- #

    def bit_regulariser(self):
        reg = 0.0
        for p in self.base_model.parameters():
            reg = reg + torch.mean(torch.abs(p))
        return reg / len(list(self.base_model.parameters()))

    def forward_with_bit_loss(self, x):
        original_params = {n: p.data.clone() for n, p in self.base_model.named_parameters()}
        with torch.no_grad():
            for p in self.base_model.parameters():
                p.data.copy_(quantize_tensor(p.data, self.bits))
        logits = self.base_model(x)
        for n, p in self.base_model.named_parameters():
            p.data.copy_(original_params[n])
        bit_loss = self.bit_regulariser()
        return logits, bit_loss

    # Mandatory for PyTorch optimizers
    def forward(self, x):
        logits, _ = self.forward_with_bit_loss(x)
        return logits


class FedMPQKDWrapper(FedMPQWrapper):
    """FedMPQ with in-round knowledge-distillation."""

    def __init__(self, base_model: nn.Module, teacher_model: nn.Module, cfg: Dict):
        super().__init__(base_model, cfg)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False


class FedPQWrapper(FedMPQWrapper):
    """Simplified implementation of FedPQ (product quantisation). Here we mimic the
    effect by uniform quantisation without the bit-regulariser."""

    def forward_with_bit_loss(self, x):
        original_params = {n: p.data.clone() for n, p in self.base_model.named_parameters()}
        with torch.no_grad():
            for p in self.base_model.parameters():
                p.data.copy_(quantize_tensor(p.data, self.bits))
        logits = self.base_model(x)
        for n, p in self.base_model.named_parameters():
            p.data.copy_(original_params[n])
        return logits, torch.tensor(0.0, device=logits.device)


class FedAvgWrapper(nn.Module):
    """Full-precision FedAvg wrapper (no quantisation, no bit loss)."""

    def __init__(self, base_model: nn.Module, cfg: Dict):
        super().__init__()
        self.base_model = base_model

    def forward_with_bit_loss(self, x):
        return self.base_model(x), torch.tensor(0.0, device=x.device)

    def forward(self, x):
        return self.base_model(x)
