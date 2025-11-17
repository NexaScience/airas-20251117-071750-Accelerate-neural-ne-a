import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .model import (
    FedAvgWrapper,
    FedMPQKDWrapper,
    FedMPQWrapper,
    FedPQWrapper,
    get_base_model,
    kd_loss,
)
from .preprocess import Preprocessor


def set_seed(seed: int):
    """Make experiments deterministic as far as possible."""

    import random
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]):
    """FedAvg aggregation (simple mean of model parameters)."""

    avg_state: Dict[str, torch.Tensor] = {}
    for k in state_dicts[0]:
        avg_state[k] = torch.stack([sd[k].float() for sd in state_dicts], 0).mean(0)
    return avg_state


def evaluate(model: torch.nn.Module, dataloader, device):
    """Return accuracy of MODEL on DATALOADER."""

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


###############################################################################
# Wrapper resolver                                                             #
###############################################################################


def _resolve_wrapper(name: str):
    name = name.lower()
    if name == "fedmpq_kd":
        return FedMPQKDWrapper
    if name == "fedmpq":
        return FedMPQWrapper
    if name == "fedpq":
        return FedPQWrapper
    if name == "fedavg":
        return FedAvgWrapper
    raise ValueError(f"Unknown model.name='{name}'. Must be one of fedavg / fedmpq / fedpq / fedmpq_kd.")


###############################################################################
# Local-training routine                                                       #
###############################################################################


def train_client(model_name: str, global_state, teacher_state, dataloader, device, cfg):
    """Single client's local optimisation step."""

    base_model = get_base_model(cfg["model"]["architecture"], cfg["data"]["num_classes"])
    base_model.load_state_dict(global_state)
    base_model = base_model.to(device)

    wrapper_cls = _resolve_wrapper(model_name)

    if wrapper_cls is FedMPQKDWrapper:
        teacher_model = get_base_model(cfg["model"]["architecture"], cfg["data"]["num_classes"])
        teacher_model.load_state_dict(teacher_state)
        teacher_model = teacher_model.to(device)
        wrapper = wrapper_cls(base_model, teacher_model, cfg)
    else:
        wrapper = wrapper_cls(base_model, cfg)

    wrapper.train()
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=cfg["training"]["lr"], momentum=0.9)

    for _ in range(cfg["training"]["local_epochs"]):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits_student, loss_bit = wrapper.forward_with_bit_loss(x)
            loss_task = F.cross_entropy(logits_student, y)

            if isinstance(wrapper, FedMPQKDWrapper):
                logits_teacher = wrapper.teacher(x)
                loss_kd = kd_loss(logits_student, logits_teacher, T=cfg["kd_params"]["T"])
                loss = (
                    loss_task
                    + cfg["model"]["lambda_b"] * loss_bit
                    + cfg["kd_params"]["alpha"] * loss_kd
                )
            else:
                loss = loss_task + cfg["model"]["lambda_b"] * loss_bit
            loss.backward()
            optimizer.step()
    return wrapper.base_model.state_dict()


###############################################################################
# Federated loop                                                               #
###############################################################################

def run_experiment(cfg: Dict, results_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data
    preproc = Preprocessor(cfg)
    client_loaders, test_loader = preproc.get_data_loaders()
    num_clients = len(client_loaders)

    # 2. Global model
    global_model = get_base_model(cfg["model"]["architecture"], cfg["data"]["num_classes"])
    global_model = global_model.to(device)

    history: List[Dict] = []

    for round_idx in range(cfg["training"]["num_rounds"]):
        per_client_states = []
        teacher_state = global_model.state_dict()  # snapshot for potential KD
        for cid, cl_loader in client_loaders.items():
            state = train_client(
                cfg["model"]["name"],
                global_model.state_dict(),
                teacher_state,
                cl_loader,
                device,
                cfg,
            )
            per_client_states.append(state)
        # Aggregate
        global_state = average_state_dicts(per_client_states)
        global_model.load_state_dict(global_state)

        # Evaluate
        acc = evaluate(global_model, test_loader, device)
        history.append({"round": round_idx + 1, "test_accuracy": acc})
        print(json.dumps({"run_id": cfg["run_id"], "round": round_idx + 1, "test_accuracy": acc}))

    # Persist
    (results_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(global_model.state_dict(), results_dir / "final_model.pt")


###############################################################################
# CLI                                                                         #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Run a single experiment variation")
    parser.add_argument("--config", required=True, help="Path to YAML config for this run")
    parser.add_argument("--results-dir", required=True, help="Directory to store artefacts of this run")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, Path(args.results_dir) / "config.yaml")

    set_seed(cfg["training"].get("seed", 0))

    print("=" * 80)
    print(f"Run-id: {cfg['run_id']}")
    print(cfg.get("description", "No description provided."))
    print("=" * 80)

    run_experiment(cfg, Path(args.results_dir))


if __name__ == "__main__":
    main()
