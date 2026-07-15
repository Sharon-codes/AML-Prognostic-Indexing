"""PyTorch 1D-CNN training and evaluation for BQPSO-selected genes."""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe for headless workstation/server execution.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)


class TranscriptomicCNN(nn.Module):
    """Two-layer 1D CNN accepting samples shaped ``(batch, genes, 1)``."""

    def __init__(self, n_genes: int) -> None:
        super().__init__()
        if n_genes < 3:
            raise ValueError("At least three selected genes are required for kernel size 3.")
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * n_genes, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Public data convention is (N, genes, 1); Conv1d expects (N, channels, length).
        x = x.transpose(1, 2)
        return self.classifier(self.features(x)).squeeze(1)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_state: int = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _save_loss_curve(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCELoss")
    plt.title("1D-CNN training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _save_roc_curve(y_true: np.ndarray, probabilities: np.ndarray, path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = float(auc(fpr, tpr))
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"CNN (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Held-out ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return roc_auc


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series, base_dir: Path, config: TrainingConfig | None = None
) -> dict[str, float]:
    """Train with an internal validation split and report held-out test metrics."""
    config = config or TrainingConfig()
    if not X.index.equals(y.index):
        raise ValueError("X and y must have matching sample indexes.")
    if y.value_counts().min() < 5:
        raise ValueError("At least five samples per class are needed for stratified train/validation/test splits.")
    _set_seed(config.random_state)
    logs_dir = base_dir / "logs_and_output"
    images_dir = base_dir / "images"
    logs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # A held-out test set is never inspected by early stopping; validation is
    # sampled only from the training partition.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32), test_size=0.20,
        stratify=y.to_numpy(), random_state=config.random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=config.random_state,
    )
    def tensorize(values: np.ndarray) -> Tensor:
        return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(tensorize(X_train), torch.tensor(y_train)), batch_size=config.batch_size, shuffle=True)
    val_x, val_y = tensorize(X_val), torch.tensor(y_val, dtype=torch.float32)
    test_x = tensorize(X_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TranscriptomicCNN(X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCELoss()
    best_state: dict[str, Tensor] | None = None
    best_val_loss = float("inf")
    no_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    LOGGER.info("Training 1D-CNN on %s (%d train / %d val / %d test).", device, len(X_train), len(X_val), len(X_test))
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(val_x.to(device)), val_y.to(device)).item())
        train_losses.append(float(np.mean(batch_losses)))
        val_losses.append(validation_loss)
        LOGGER.info("CNN epoch %03d | train loss %.5f | val loss %.5f", epoch, train_losses[-1], validation_loss)
        if validation_loss < best_val_loss - 1e-6:
            best_val_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= config.patience:
                LOGGER.info("Early stopping at epoch %d (patience=%d).", epoch, config.patience)
                break
    if best_state is None:
        raise RuntimeError("CNN training did not produce a valid model state.")
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "selected_genes": list(X.columns), "config": vars(config)}, logs_dir / "best_model.pth")
    _save_loss_curve(train_losses, val_losses, images_dir / "training_loss_curve.png")

    model.eval()
    with torch.no_grad():
        probabilities = model(test_x.to(device)).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(np.int8)
    y_test_int = y_test.astype(np.int8)
    roc_auc = _save_roc_curve(y_test_int, probabilities, images_dir / "roc_curve.png")
    metrics = {
        "accuracy": float(accuracy_score(y_test_int, predictions)),
        "precision": float(precision_score(y_test_int, predictions, zero_division=0)),
        "recall": float(recall_score(y_test_int, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test_int, predictions, zero_division=0)),
        "auc": roc_auc,
    }
    pd.DataFrame([metrics]).to_csv(logs_dir / "cnn_metrics.csv", index=False)
    LOGGER.info("Held-out CNN metrics: %s", ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
    return metrics


if __name__ == "__main__":
    raise SystemExit("Run main.py to execute the complete pipeline.")
