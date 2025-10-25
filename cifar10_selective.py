
"""
CIFAR-10 Selective Classification Utilities (Step 2)
----------------------------------------------------
This module provides:
  - Data loading with train/val/calibration/test splits
  - A small ResNet18 model head for CIFAR-10
  - Training & evaluation loops
  - Inference utilities to get f(x) = logits
  - Selection score utilities g(x): MSP, margin, entropy, energy, and temperature scaling

Usage (CLI quickstart):
  python cifar10_selective.py --data_root ./data --epochs 5 --batch_size 256 --device cuda

Outputs:
  - logits and scores saved to .npz files if --save_npz is provided

Notes:
  - Keep the calibration split separate from validation.
  - This is a lightweight, reproducible scaffold to plug into Algorithm 1 next.
"""
import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

try:
    from sklearn.model_selection import StratifiedShuffleSplit
except Exception:
    StratifiedShuffleSplit = None
    print("[Warning] scikit-learn not found. Falling back to non-stratified splits. Install scikit-learn for stratified splits.")

# --------------------
# Configs & Utilities
# --------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

@dataclass
class TrainConfig:
    data_root: str = "./data"
    batch_size: int = 256
    epochs: int = 5
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    seed: int = 0
    val_ratio: float = 0.1
    cal_ratio: float = 0.2
    num_workers: int = 4
    device: str = "cuda"
    save_npz: Optional[str] = None
    temperature_init: float = 1.0
    temperature_lr: float = 0.01
    temperature_epochs: int = 50

def set_seed(seed: int):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------
# Data
# --------------------

def _transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

def _split_indices_stratified(y: np.ndarray, val_ratio: float, cal_ratio: float, seed: int):
    n = len(y)
    all_idx = np.arange(n)

    # First: split out validation
    if StratifiedShuffleSplit is not None:
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss_val.split(all_idx, y))
    else:
        # Non-stratified fallback
        rng = np.random.default_rng(seed)
        rng.shuffle(all_idx)
        cut = int((1 - val_ratio) * n)
        train_idx, val_idx = all_idx[:cut], all_idx[cut:]

    # Then: split calibration from the remaining train pool
    remaining_y = y[train_idx]
    if StratifiedShuffleSplit is not None:
        cal_rel = cal_ratio / max(1e-9, (1 - val_ratio))  # proportion among remaining
        sss_cal = StratifiedShuffleSplit(n_splits=1, test_size=cal_rel, random_state=seed + 1)
        train_idx2, cal_idx_rel = next(sss_cal.split(train_idx, remaining_y))
        train_idx, cal_idx = train_idx[train_idx2], train_idx[cal_idx_rel]
    else:
        rng = np.random.default_rng(seed + 1)
        rng.shuffle(train_idx)
        cal_count = int(cal_ratio * n)
        cal_idx, train_idx = train_idx[:cal_count], train_idx[cal_count:]

    return train_idx, val_idx, cal_idx

def get_cifar10_dataloaders(
    data_root: str,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    cal_ratio: float = 0.2,
    seed: int = 0,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    # Two copies of the training set with different transforms
    train_aug = datasets.CIFAR10(root=data_root, train=True, transform=_transforms(train=True), download=True)
    train_plain = datasets.CIFAR10(root=data_root, train=True, transform=_transforms(train=False), download=True)
    y = np.array(train_aug.targets)

    train_idx, val_idx, cal_idx = _split_indices_stratified(y, val_ratio, cal_ratio, seed)

    ds_train = Subset(train_aug, train_idx)
    ds_val = Subset(train_plain, val_idx)
    ds_cal = Subset(train_plain, cal_idx)
    ds_test = datasets.CIFAR10(root=data_root, train=False, transform=_transforms(train=False), download=True)

    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "cal":   DataLoader(ds_cal,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders

# --------------------
# Model
# --------------------

def create_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    # Torchvision API compatibility (pretrained vs weights)
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# --------------------
# Training & Eval
# --------------------

def evaluate(model: nn.Module, loader: DataLoader, device: str = "cuda") -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0.0, 0, 0
    device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_count += x.size(0)
    return total_loss / total_count, total_correct / total_count

def train_one_epoch(model, loader, opt, device="cuda"):
    model.train()
    ce = nn.CrossEntropyLoss()
    device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def train_model(model: nn.Module,
                loaders: Dict[str, DataLoader],
                cfg: TrainConfig) -> nn.Module:
    device = cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], opt, device=device)
        val_loss, val_acc = evaluate(model, loaders["val"], device=device)
        scheduler.step()

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device)

# --------------------
# Inference: f(x) logits
# --------------------

@torch.no_grad()
def get_logits(model: nn.Module, loader: DataLoader, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    all_logits = []
    all_y = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)

# --------------------
# Selection scores: g(x)
# --------------------

def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(1e-8, temperature)
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def compute_scores_from_logits(logits: np.ndarray, temperature: float = 1.0):
    """
    Returns a dict of selection scores:
      - msp: max softmax probability
      - margin: p_top1 - p_top2
      - entropy: -sum p log p
      - energy: logsumexp(logits / T)
    """
    probs = softmax_np(logits, temperature=temperature)
    sort_probs = np.sort(probs, axis=1)[:, ::-1]
    top1 = sort_probs[:, 0]
    top2 = sort_probs[:, 1] if sort_probs.shape[1] > 1 else np.zeros_like(top1)
    entropy = -(probs * (np.log(probs + 1e-12))).sum(axis=1)

    scaled = logits / max(1e-8, temperature)
    maxz = scaled.max(axis=1, keepdims=True)
    energy = (maxz + np.log(np.exp(scaled - maxz).sum(axis=1, keepdims=True))).squeeze(1)

    return {
        "msp": top1,
        "margin": top1 - top2,
        "entropy": entropy,
        "energy": energy,
    }

# --------------------
# Temperature scaling (optional)
# --------------------

class TemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(math.log(init_temp), dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = torch.exp(self.log_t)
        return logits / t

    def temperature(self) -> float:
        return float(torch.exp(self.log_t).item())

def fit_temperature(model: nn.Module,
                    loader: DataLoader,
                    init_temp: float = 1.0,
                    lr: float = 0.01,
                    epochs: int = 50,
                    device: str = "cuda") -> float:
    device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    model.eval()
    scaler = TemperatureScaler(init_temp).to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=epochs)
    ce = nn.CrossEntropyLoss()

    # collect logits and labels for calibration
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            logits_list.append(logits)
            labels_list.append(y.to(device))
    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    def closure():
        opt.zero_grad()
        scaled = scaler(logits_all)
        loss = ce(scaled, labels_all)
        loss.backward()
        return loss

    opt.step(closure)
    return scaler.temperature()

# --------------------
# Saving
# --------------------

def save_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[Saved] {path}")

# --------------------
# CLI
# --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--cal_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_npz", type=str, default=None, help="Path to save logits/scores (e.g., ./artifacts/cifar10_scores.npz)")
    parser.add_argument("--fit_temperature", action="store_true", help="Fit temperature on calibration split before scoring.")
    parser.add_argument("--temperature_init", type=float, default=1.0)
    parser.add_argument("--temperature_lr", type=float, default=0.01)
    parser.add_argument("--temperature_epochs", type=int, default=50)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        seed=args.seed,
        val_ratio=args.val_ratio,
        cal_ratio=args.cal_ratio,
        num_workers=args.num_workers,
        device=args.device,
        save_npz=args.save_npz,
        temperature_init=args.temperature_init,
        temperature_lr=args.temperature_lr,
        temperature_epochs=args.temperature_epochs,
    )

    set_seed(cfg.seed)
    loaders = get_cifar10_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        cal_ratio=cfg.cal_ratio,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )

    model = create_resnet18_cifar(num_classes=10)
    device = cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    model = train_model(model, loaders, cfg)

    # Evaluate on val/test for a quick check
    val_loss, val_acc = evaluate(model, loaders["val"], device=device)
    test_loss, test_acc = evaluate(model, loaders["test"], device=device)
    print(f"[Val]  loss={val_loss:.4f}  acc={val_acc:.4f}")
    print(f"[Test] loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Optionally fit temperature on calibration split
    temperature = 1.0
    if args.fit_temperature:
        temperature = fit_temperature(model, loaders["cal"],
                                      init_temp=cfg.temperature_init,
                                      lr=cfg.temperature_lr,
                                      epochs=cfg.temperature_epochs,
                                      device=device)
        print(f"[Temperature] Fitted temperature: {temperature:.4f}")

    # Produce logits and scores for val/cal/test
    out = {}
    for split in ["val", "cal", "test"]:
        logits, y = get_logits(model, loaders[split], device=device)
        scores = compute_scores_from_logits(logits, temperature=temperature)
        out[f"logits_{split}"] = logits
        out[f"y_{split}"] = y
        for k, v in scores.items():
            out[f"{k}_{split}"] = v

    if cfg.save_npz:
        save_npz(cfg.save_npz, **out)

if __name__ == "__main__":
    main()
