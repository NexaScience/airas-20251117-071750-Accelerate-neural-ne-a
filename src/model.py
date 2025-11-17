"""Model architectures and wrappers for FedMPQ & FedMPQ-KD."""
from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm  # Lightweight, pure-python model zoo
except ImportError:  # pragma: no cover
    timm = None

###############################################################################
# KD loss                                                                      #
###############################################################################

def kd_loss(logits_student, logits_teacher, T: float = 2.0):
    p_s = F.log_softmax(logits_student / T, dim=1)
    p_t = F.softmax(logits_teacher.detach() / T, dim=1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)


###############################################################################
# Quantisation helpers                                                         #
###############################################################################

def quantize_tensor(t: torch.Tensor, num_bits: int = 8):
    """Symmetric uniform fake-quantisation (per-tensor)."""
    if num_bits >= 32:
        return t  # No quantisation

    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    min_val, max_val = t.min(), t.max()
    # Avoid degenerate scale
    scale = (max_val - min_val) / float(qmax - qmin) if max_val != min_val else torch.tensor(1.0, device=t.device)
    zp = qmin - torch.round(min_val / scale)
    qt = torch.clamp(torch.round(t / scale + zp), qmin, qmax)
    return (qt - zp) * scale


###############################################################################
# Tiny CNN for smoke tests                                                     #
###############################################################################


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


###############################################################################
# ResNet-20 (CIFAR)                                                            #
###############################################################################


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
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


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


def ResNet20(num_classes=10):
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


###############################################################################
# MobileNet-V2 (via timm)                                                      #
###############################################################################


def MobileNetV2(num_classes=10):
    if timm is None:
        raise ImportError("timm is required for MobileNetV2 – please install timm>=0.9.12")
    return timm.create_model("mobilenetv2_100.ra_in1k", pretrained=False, num_classes=num_classes)


###############################################################################
# Model factory                                                                #
###############################################################################

BASE_MODEL_FACTORY = {
    "tiny_cnn": TinyCNN,
    "resnet20": ResNet20,
    "mobilenetv2": MobileNetV2,
}


def get_base_model(name: str, num_classes: int):
    name = name.lower()
    if name not in BASE_MODEL_FACTORY:
        raise NotImplementedError(f"Model architecture '{name}' not implemented.")
    return BASE_MODEL_FACTORY[name](num_classes=num_classes)


###############################################################################
# Wrappers                                                                     #
###############################################################################


class FedMPQWrapper(nn.Module):
    """Wraps a base model with (fake) quantisation + bit regulariser."""

    def __init__(self, base_model: nn.Module, cfg: Dict):
        super().__init__()
        self.base_model = base_model
        self.bits = cfg["model"].get("bits", 32)
        self.lambda_b = cfg["model"].get("lambda_b", 0.0)

    # --------------------------------------------------------
    def forward_with_bit_loss(self, x):
        # Fake quantise weights (in-place) then restore
        original_params = {n: p.data.clone() for n, p in self.base_model.named_parameters()}
        with torch.no_grad():
            for p in self.base_model.parameters():
                p.data.copy_(quantize_tensor(p.data, self.bits))
        logits = self.base_model(x)
        # Restore fp32 weights for gradient flow
        for n, p in self.base_model.named_parameters():
            p.data.copy_(original_params[n])

        bit_loss = self.bit_regulariser() if self.lambda_b > 0 else torch.tensor(0.0, device=x.device)
        return logits, bit_loss

    # --------------------------------------------------------
    def forward(self, x):
        logits, _ = self.forward_with_bit_loss(x)
        return logits

    # --------------------------------------------------------
    def bit_regulariser(self):
        reg = 0.0
        for p in self.base_model.parameters():
            reg += torch.mean(torch.abs(p))
        return reg


class FedMPQKDWrapper(FedMPQWrapper):
    """Extends FedMPQ with knowledge-distillation to a frozen teacher."""

    def __init__(self, base_model: nn.Module, teacher_model: nn.Module, cfg: Dict):
        super().__init__(base_model, cfg)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
