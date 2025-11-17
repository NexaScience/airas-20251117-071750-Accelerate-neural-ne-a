import argparse
import json
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
    FedMPQKDWrapper,
    FedMPQWrapper,
    get_base_model,
    kd_loss,
)
from .preprocess import Preprocessor


################################################################################
# Helper functions                                                             #
################################################################################

def set_seed(seed: int):
    """Make experiments deterministic (as much as possible)."""
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
    avg_state = {}
    for k in state_dicts[0]:
        avg_state[k] = (
            torch.stack([sd[k].float() for sd in state_dicts], 0).mean(0).type(state_dicts[0][k].dtype)
        )
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


################################################################################
# Client training                                                             #
################################################################################

def _make_optimizer(params, cfg):
    opt_name = cfg["training"].get("optimizer", "sgd").lower()
    lr = cfg["training"]["lr"]
    if opt_name == "adamw":
        weight_decay = cfg["training"].get("weight_decay", 1e-4)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # Default: SGD
    momentum = cfg["training"].get("momentum", 0.9)
    weight_decay = cfg["training"].get("weight_decay", 5e-4)
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def train_client(
    model_wrapper_cls,
    global_state,
    teacher_state,
    dataloader,
    device,
    cfg,
):
    """Single client's local training step."""
    # Instantiate student
    base_model = get_base_model(cfg["model"]["architecture"], cfg["data"]["num_classes"])
    base_model.load_state_dict(global_state)
    base_model.to(device)

    # Teacher (only for KD-style wrappers)
    if model_wrapper_cls is FedMPQKDWrapper:
        teacher_model = get_base_model(cfg["model"]["architecture"], cfg["data"]["num_classes"])
        teacher_model.load_state_dict(teacher_state)
        teacher_model.to(device)
        wrapper = model_wrapper_cls(base_model, teacher_model, cfg)
    else:
        wrapper = model_wrapper_cls(base_model, cfg)

    wrapper.train()
    optimizer = _make_optimizer(wrapper.parameters(), cfg)

    kd_cfg = cfg.get("kd_params", {"alpha": 0.5, "T": 2.0})

    for _ in range(cfg["training"]["local_epochs"]):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits_student, loss_bit = wrapper.forward_with_bit_loss(x)
            loss_task = F.cross_entropy(logits_student, y)
            if model_wrapper_cls is FedMPQKDWrapper:
                with torch.no_grad():
                    logits_teacher = wrapper.teacher(x)
                loss_kd = kd_loss(logits_student, logits_teacher, T=kd_cfg["T"])
                loss = (
                    loss_task
                    + cfg["model"].get("lambda_b", 0.0) * loss_bit
                    + kd_cfg["alpha"] * loss_kd
                )
            else:
                loss = loss_task + cfg["model"].get("lambda_b", 0.0) * loss_bit
            loss.backward()
            optimizer.step()
    # Return student weights
    return wrapper.base_model.state_dict()


################################################################################
# Federated loop                                                              #
################################################################################

def run_experiment(cfg: Dict, results_dir: Path):
    """Main federated training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data
    preproc = Preprocessor(cfg)
    client_loaders, test_loader = preproc.get_data_loaders()
    num_clients = len(client_loaders)
    participation_rate = cfg["dataset"].get("participation_rate", 1.0)

    # 2. Global model
    global_model = get_base_model(cfg["model"]["architecture"], cfg["data"]["num_classes"])
    global_model.to(device)

    # 3. Training history
    history = []

    for round_idx in range(cfg["training"]["num_rounds"]):
        round_client_states = []
        teacher_state = global_model.state_dict()

        # Sample active clients
        num_active = max(1, int(num_clients * participation_rate))
        active_clients = np.random.choice(list(client_loaders.keys()), num_active, replace=False)

        for cid in active_clients:
            # 20% chance of dropout (straggler) if configured
            dropout_prob = cfg["dataset"].get("dropout_prob", 0.0)
            if np.random.rand() < dropout_prob:
                continue  # skip this client

            cl_loader = client_loaders[cid]
            wrapper_cls = (
                FedMPQKDWrapper if "kd" in cfg["model"]["name"].lower() else FedMPQWrapper
            )
            client_state = train_client(
                wrapper_cls,
                global_model.state_dict(),
                teacher_state,
                cl_loader,
                device,
                cfg,
            )
            round_client_states.append(client_state)

        # Guard against all clients dropped out
        if not round_client_states:
            print(f"[WARN] All clients dropped in round {round_idx + 1}. Skipping aggregation.")
            continue

        # Aggregate
        new_global_state = average_state_dicts(round_client_states)
        global_model.load_state_dict(new_global_state)

        # Evaluate
        acc = evaluate(global_model, test_loader, device)
        history.append({"round": round_idx + 1, "test_accuracy": acc})
        print(json.dumps({"run_id": cfg["run_id"], "round": round_idx + 1, "test_accuracy": acc}))

    # Persist results
    results_path = results_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(history, f, indent=2)

    torch.save(global_model.state_dict(), results_dir / "final_model.pt")


################################################################################
# CLI                                                                         #
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Run a single experiment variation (client-server FL loop)"
    )
    parser.add_argument("--config", required=True, help="Path to variation YAML config")
    parser.add_argument("--results-dir", required=True, help="Where to store outputs for this run")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Echo description
    print("=" * 80)
    print(f"Experiment description (run_id = {cfg['run_id']}):\n{cfg.get('description', '')}")
    print("=" * 80)

    # Copy config for reproducibility
    shutil.copy(args.config, Path(args.results_dir) / "config.yaml")

    set_seed(cfg["training"].get("seed", 0))
    start = time.time()
    run_experiment(cfg, Path(args.results_dir))
    print(f"Run {cfg['run_id']} completed in {(time.time() - start)/60:.2f} minutes")


if __name__ == "__main__":
    main()
