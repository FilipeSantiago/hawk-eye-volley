from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

InputMode = Literal["rgb", "brightness", "rgb_brightness"]
VALID_INPUT_MODES = ("rgb", "brightness", "rgb_brightness")


def get_input_channels(input_mode: InputMode) -> int:
    if input_mode == "rgb":
        return 3
    if input_mode == "brightness":
        return 1
    if input_mode == "rgb_brightness":
        return 4
    raise ValueError(
        f"Unsupported input_mode={input_mode!r}. "
        f"Expected one of {VALID_INPUT_MODES}."
    )


def infer_binary_label_from_path(npz_path: Path) -> float:
    parts = set(npz_path.parts)
    if "_ball not" in parts:
        return 0.0
    if "_ball" in parts:
        return 1.0
    raise ValueError(
        f"Could not infer label from path {npz_path}. "
        "Expected the path to include either '_ball' or '_ball not'."
    )


def _collect_npz_files(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        print(f"[WARN] Dataset directory does not exist: {root_dir}")
        return []
    return sorted(path for path in root_dir.rglob("*.npz") if path.is_file())


@dataclass(frozen=True)
class NpzSplitPaths:
    name: str
    ball_paths: List[Path]
    no_ball_paths: List[Path]

    @property
    def all_paths(self) -> List[Path]:
        return [*self.ball_paths, *self.no_ball_paths]

    @property
    def total_count(self) -> int:
        return len(self.ball_paths) + len(self.no_ball_paths)


def collect_split_paths(
    ball_dir: Path,
    no_ball_dir: Path,
    split_name: str,
) -> NpzSplitPaths:
    ball_paths = _collect_npz_files(ball_dir)
    no_ball_paths = _collect_npz_files(no_ball_dir)

    print(
        f"{split_name}: "
        f"ball={len(ball_paths)} "
        f"no_ball={len(no_ball_paths)} "
        f"total={len(ball_paths) + len(no_ball_paths)}"
    )

    return NpzSplitPaths(
        name=split_name,
        ball_paths=ball_paths,
        no_ball_paths=no_ball_paths,
    )


def collect_train_val_npz_paths(
    train_ball_dir: Path,
    val_ball_dir: Path,
    train_no_ball_dir: Path,
    val_no_ball_dir: Path,
) -> Tuple[NpzSplitPaths, NpzSplitPaths]:
    train_split = collect_split_paths(
        ball_dir=train_ball_dir,
        no_ball_dir=train_no_ball_dir,
        split_name="train",
    )
    val_split = collect_split_paths(
        ball_dir=val_ball_dir,
        no_ball_dir=val_no_ball_dir,
        split_name="val",
    )
    return train_split, val_split


class BallCandidateNpzDataset(Dataset):
    def __init__(
        self,
        npz_paths: Sequence[Path],
        input_mode: InputMode = "rgb_brightness",
        check_embedded_label: bool = True,
    ) -> None:
        self.npz_paths = [Path(path) for path in npz_paths]
        self.input_mode = input_mode
        self.check_embedded_label = check_embedded_label
        self.in_channels = get_input_channels(input_mode)

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        npz_path = self.npz_paths[index]
        folder_label = infer_binary_label_from_path(npz_path)

        try:
            with np.load(npz_path, allow_pickle=False) as data:
                image = np.asarray(data["image"])
                brightness = np.asarray(data["brightness"])
                embedded_label = data["label"] if "label" in data.files else None
        except Exception as exc:
            raise RuntimeError(f"Failed to load NPZ sample: {npz_path} ({exc})") from exc

        self._validate_sample(npz_path=npz_path, image=image, brightness=brightness)
        self._check_label_consistency(
            npz_path=npz_path,
            folder_label=folder_label,
            embedded_label=embedded_label,
        )

        image_tensor = image.astype(np.float32) / 255.0
        brightness_tensor = brightness.astype(np.float32) / 255.0

        rgb_channels = np.transpose(image_tensor, (2, 0, 1))
        brightness_channel = brightness_tensor[None, :, :]

        if self.input_mode == "rgb":
            x = rgb_channels
        elif self.input_mode == "brightness":
            x = brightness_channel
        elif self.input_mode == "rgb_brightness":
            x = np.concatenate([rgb_channels, brightness_channel], axis=0)
        else:
            raise ValueError(
                f"Unsupported input_mode={self.input_mode!r}. "
                f"Expected one of {VALID_INPUT_MODES}."
            )

        x_tensor = torch.from_numpy(np.ascontiguousarray(x)).to(dtype=torch.float32)
        y_tensor = torch.tensor(folder_label, dtype=torch.float32)
        return x_tensor, y_tensor

    def _check_label_consistency(
        self,
        npz_path: Path,
        folder_label: float,
        embedded_label: Optional[np.ndarray],
    ) -> None:
        if not self.check_embedded_label or embedded_label is None:
            return

        embedded_value = float(np.asarray(embedded_label).item())
        if embedded_value != folder_label:
            raise ValueError(
                f"Label mismatch for {npz_path}: "
                f"folder label={folder_label}, embedded label={embedded_value}."
            )

    @staticmethod
    def _validate_sample(
        npz_path: Path,
        image: np.ndarray,
        brightness: np.ndarray,
    ) -> None:
        if image.shape != (64, 64, 3):
            raise ValueError(
                f"Invalid image shape for {npz_path}: "
                f"expected (64, 64, 3), got {image.shape}."
            )
        if brightness.shape != (64, 64):
            raise ValueError(
                f"Invalid brightness shape for {npz_path}: "
                f"expected (64, 64), got {brightness.shape}."
            )


def build_dataloaders(
    train_paths: Sequence[Path],
    val_paths: Sequence[Path],
    input_mode: InputMode,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset = BallCandidateNpzDataset(
        npz_paths=train_paths,
        input_mode=input_mode,
    )
    val_dataset = BallCandidateNpzDataset(
        npz_paths=val_paths,
        input_mode=input_mode,
    )

    common_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader
