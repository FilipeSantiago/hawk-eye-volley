from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ProcessedCandidateSample:
    image: np.ndarray  # RGB, shape (H, W, 3)
    brightness: np.ndarray  # grayscale, shape (H, W)
    source_path: Path
    frame_stem: str
    candidate_index: int  # 0-based
    bbox_xywh: tuple[int, int, int, int]
    square_crop_xywh: tuple[int, int, int, int]

