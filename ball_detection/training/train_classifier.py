from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ball_detection.data.npz_dataset import (
    BallCandidateNpzDataset,
    NpzSplitPaths,
    build_dataloaders,
    collect_train_val_npz_paths,
    get_input_channels,
)
from ball_detection.training.metrics import BinaryClassificationMetrics, compute_binary_metrics_from_logits


class BallCandidateCNN(nn.Module):
    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)


def _prepare_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim == 2:
        if logits.shape[1] != 1:
            raise ValueError(
                f"Expected model output with shape [B] or [B, 1], got {tuple(logits.shape)}."
            )
        logits = logits[:, 0]
    elif logits.ndim != 1:
        raise ValueError(
            f"Expected model output with shape [B] or [B, 1], got {tuple(logits.shape)}."
        )

    targets = targets.reshape(-1).to(dtype=torch.float32)
    logits = logits.reshape(-1)

    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Batch size mismatch between logits and targets: "
            f"{tuple(logits.shape)} vs {tuple(targets.shape)}."
        )

    return logits, targets


def _resolve_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: Optional[torch.device] = None,
) -> float:
    resolved_device = _resolve_device(device)
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(resolved_device, non_blocking=True)
        y = y.to(resolved_device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        logits, targets = _prepare_logits_and_targets(logits, y)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    resolved_device = _resolve_device(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(resolved_device, non_blocking=True)
            y = y.to(resolved_device, non_blocking=True)

            logits = model(x)
            logits, targets = _prepare_logits_and_targets(logits, y)

            loss = criterion(logits, targets)
            batch_size = targets.shape[0]

            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    val_loss = total_loss / max(total_samples, 1)

    if not all_logits:
        empty_metrics = BinaryClassificationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            tp=0,
            tn=0,
            fp=0,
            fn=0,
            threshold=threshold,
        )
        metrics_dict = empty_metrics.to_dict()
    else:
        metrics = compute_binary_metrics_from_logits(
            logits=torch.cat(all_logits, dim=0),
            targets=torch.cat(all_targets, dim=0),
            threshold=threshold,
        )
        metrics_dict = metrics.to_dict()

    metrics_dict["loss"] = val_loss
    return metrics_dict


def _is_better_checkpoint(
    best_record: Optional[Dict[str, object]],
    candidate_record: Dict[str, object],
) -> bool:
    if best_record is None:
        return True

    best_f1 = float(best_record["val_f1"])
    candidate_f1 = float(candidate_record["val_f1"])
    if candidate_f1 > best_f1:
        return True
    if candidate_f1 == best_f1 and float(candidate_record["val_loss"]) < float(
        best_record["val_loss"]
    ):
        return True
    return False


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    model_save_path: Path = Path("ball_cnn_best.pt"),
    device: Optional[torch.device] = None,
) -> List[Dict[str, object]]:
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, object]] = []
    best_record: Optional[Dict[str, object]] = None
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {resolved_device}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=resolved_device,
        )
        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            threshold=threshold,
            device=resolved_device,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "tp": int(val_metrics["tp"]),
            "tn": int(val_metrics["tn"]),
            "fp": int(val_metrics["fp"]),
            "fn": int(val_metrics["fn"]),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"val_loss={epoch_record['val_loss']:.4f} "
            f"val_acc={epoch_record['val_accuracy']:.4f} "
            f"val_precision={epoch_record['val_precision']:.4f} "
            f"val_recall={epoch_record['val_recall']:.4f} "
            f"val_f1={epoch_record['val_f1']:.4f} "
            f"TP={epoch_record['tp']} "
            f"TN={epoch_record['tn']} "
            f"FP={epoch_record['fp']} "
            f"FN={epoch_record['fn']}"
        )

        if _is_better_checkpoint(best_record=best_record, candidate_record=epoch_record):
            best_record = epoch_record
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_record["train_loss"],
                    "val_loss": epoch_record["val_loss"],
                    "val_metrics": val_metrics,
                    "input_mode": getattr(train_loader.dataset, "input_mode", None),
                    "in_channels": getattr(train_loader.dataset, "in_channels", None),
                    "threshold": threshold,
                },
                model_save_path,
            )
            print(f"Saved best model to {model_save_path}")

    return history


def run_dataset_sanity_checks(
    train_split: NpzSplitPaths,
    val_split: NpzSplitPaths,
    input_mode: str,
) -> Dict[str, object]:
    expected_channels = get_input_channels(input_mode)

    for split in (train_split, val_split):
        if not split.ball_paths:
            raise AssertionError(f"{split.name} split has no ball samples.")
        if not split.no_ball_paths:
            raise AssertionError(f"{split.name} split has no no_ball samples.")

    train_ball_x, train_ball_y = BallCandidateNpzDataset(
        [train_split.ball_paths[0]],
        input_mode=input_mode,
    )[0]
    train_no_ball_x, train_no_ball_y = BallCandidateNpzDataset(
        [train_split.no_ball_paths[0]],
        input_mode=input_mode,
    )[0]
    val_ball_x, val_ball_y = BallCandidateNpzDataset(
        [val_split.ball_paths[0]],
        input_mode=input_mode,
    )[0]
    val_no_ball_x, val_no_ball_y = BallCandidateNpzDataset(
        [val_split.no_ball_paths[0]],
        input_mode=input_mode,
    )[0]

    for x_tensor in (train_ball_x, train_no_ball_x, val_ball_x, val_no_ball_x):
        if x_tensor.dtype != torch.float32:
            raise AssertionError(f"Expected float32 inputs, got {x_tensor.dtype}.")
        if tuple(x_tensor.shape) != (expected_channels, 64, 64):
            raise AssertionError(
                f"Expected shape {(expected_channels, 64, 64)}, got {tuple(x_tensor.shape)}."
            )

    for y_tensor, expected_label in (
        (train_ball_y, 1.0),
        (train_no_ball_y, 0.0),
        (val_ball_y, 1.0),
        (val_no_ball_y, 0.0),
    ):
        if y_tensor.dtype != torch.float32:
            raise AssertionError(f"Expected float32 labels, got {y_tensor.dtype}.")
        if y_tensor.ndim != 0:
            raise AssertionError(f"Expected scalar label tensor, got shape {tuple(y_tensor.shape)}.")
        if float(y_tensor.item()) != expected_label:
            raise AssertionError(
                f"Expected label {expected_label}, got {float(y_tensor.item())}."
            )

    batch_dataset = BallCandidateNpzDataset(
        [
            train_split.ball_paths[0],
            train_split.no_ball_paths[0],
        ],
        input_mode=input_mode,
    )
    batch_x = torch.stack([batch_dataset[0][0], batch_dataset[1][0]], dim=0)
    batch_y = torch.stack([batch_dataset[0][1], batch_dataset[1][1]], dim=0)
    dummy_logits = torch.zeros(batch_y.shape[0], dtype=torch.float32)
    nn.BCEWithLogitsLoss()(dummy_logits, batch_y)

    summary = {
        "train_ball_count": len(train_split.ball_paths),
        "train_no_ball_count": len(train_split.no_ball_paths),
        "val_ball_count": len(val_split.ball_paths),
        "val_no_ball_count": len(val_split.no_ball_paths),
        "input_mode": input_mode,
        "in_channels": expected_channels,
        "sample_shape": tuple(train_ball_x.shape),
    }

    print("Sanity checks passed:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return summary


INPUT_MODE = "rgb_brightness"
TRAIN_BALL_DIR = Path("/home/skynet/Downloads/hawk eye/data/_ball/npz/train")
VAL_BALL_DIR = Path("/home/skynet/Downloads/hawk eye/data/_ball/npz/val")
TRAIN_NO_BALL_DIR = Path("/home/skynet/Downloads/hawk eye/data/_ball not/npz/train")
VAL_NO_BALL_DIR = Path("/home/skynet/Downloads/hawk eye/data/_ball not/npz/val")
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
MODEL_SAVE_PATH = Path("models") / "ball_cnn_best.pt"


def main() -> List[Dict[str, object]]:
    train_split, val_split = collect_train_val_npz_paths(
        train_ball_dir=TRAIN_BALL_DIR,
        val_ball_dir=VAL_BALL_DIR,
        train_no_ball_dir=TRAIN_NO_BALL_DIR,
        val_no_ball_dir=VAL_NO_BALL_DIR,
    )
    run_dataset_sanity_checks(
        train_split=train_split,
        val_split=val_split,
        input_mode=INPUT_MODE,
    )

    train_loader, val_loader = build_dataloaders(
        train_paths=train_split.all_paths,
        val_paths=val_split.all_paths,
        input_mode=INPUT_MODE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = BallCandidateCNN(in_channels=get_input_channels(INPUT_MODE))
    return fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        threshold=THRESHOLD,
        model_save_path=MODEL_SAVE_PATH,
    )


if __name__ == "__main__":
    main()
