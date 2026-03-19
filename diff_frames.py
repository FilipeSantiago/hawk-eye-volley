from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DiffConfig:
    # Main config
    input_frames_dir: str
    output_diff_dir: str
    start_frame: int = 2  # 1-based frame number, needs a previous frame
    end_frame: Optional[int] = None  # 1-based inclusive, None = last frame
    use_roi: bool = False
    save_thresholded: bool = False
    diff_threshold: int = 12
    normalize_diff: bool = False
    blur_ksize: Tuple[int, int] = (3, 3)
    output_ext: str = ".png"

    # Simple ROI mask controls (easy to edit)
    roi_bottom_strip_pct: float = 0.08
    roi_watermark_w_pct: float = 0.14
    roi_watermark_h_pct: float = 0.12
    roi_watermark_corner: str = "bottom_right"

    # Logging
    progress_log_every: int = 100


CONFIG = DiffConfig(
    input_frames_dir="/home/bugslayer/Downloads/volley video footage/frames/videoplayback-00.06.05.191-00.11.14.198",
    output_diff_dir="/home/bugslayer/Downloads/volley video footage/diff_frames",
    start_frame=2,
    end_frame=None,
    use_roi=False,
    save_thresholded=False,
    diff_threshold=12,
    normalize_diff=False,
    blur_ksize=(3, 3),
    output_ext=".png",
    progress_log_every=200,
)


def iter_images(input_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def preprocess_gray(frame: np.ndarray, blur_ksize: Tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_ksize[0] > 0 and blur_ksize[1] > 0:
        gray = cv2.GaussianBlur(gray, blur_ksize, 0)
    return gray


def make_roi_mask(frame_shape: Tuple[int, int, int], config: DiffConfig) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)

    bottom_strip = int(h * config.roi_bottom_strip_pct)
    if bottom_strip > 0:
        mask[h - bottom_strip : h, :] = 0

    wm_w = int(w * config.roi_watermark_w_pct)
    wm_h = int(h * config.roi_watermark_h_pct)
    if wm_w > 0 and wm_h > 0:
        corner = config.roi_watermark_corner.lower()
        if corner == "bottom_left":
            x0, y0 = 0, h - wm_h
        elif corner == "top_left":
            x0, y0 = 0, 0
        elif corner == "top_right":
            x0, y0 = w - wm_w, 0
        else:
            x0, y0 = w - wm_w, h - wm_h
        x1, y1 = min(w, x0 + wm_w), min(h, y0 + wm_h)
        mask[y0:y1, x0:x1] = 0

    return mask


def compute_diff(prev_frame: np.ndarray, curr_frame: np.ndarray, config: DiffConfig) -> np.ndarray:
    prev_gray = preprocess_gray(prev_frame, config.blur_ksize)
    curr_gray = preprocess_gray(curr_frame, config.blur_ksize)
    diff = cv2.absdiff(curr_gray, prev_gray)
    return diff


def process_frames(config: DiffConfig) -> None:
    input_dir = Path(config.input_frames_dir)
    output_dir = Path(config.output_diff_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    start_idx = max(config.start_frame - 1, 1)
    end_idx = (len(images) - 1) if config.end_frame is None else min(config.end_frame - 1, len(images) - 1)
    if end_idx < start_idx:
        raise ValueError("end_frame must be >= start_frame, and start_frame must be >= 2.")

    first_frame = cv2.imread(str(images[start_idx]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {images[start_idx]}")

    roi_mask = make_roi_mask(first_frame.shape, config) if config.use_roi else None
    thr_dir = None
    if config.save_thresholded:
        thr_dir = output_dir / "thresholded"
        thr_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for idx in range(start_idx, end_idx + 1):
        prev_path = images[idx - 1]
        curr_path = images[idx]

        prev = cv2.imread(str(prev_path))
        curr = cv2.imread(str(curr_path))
        if prev is None or curr is None:
            continue

        diff = compute_diff(prev, curr, config)
        if roi_mask is not None:
            diff = cv2.bitwise_and(diff, roi_mask)

        if config.normalize_diff:
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        out_name = f"diff_{curr_path.stem}{config.output_ext}"
        cv2.imwrite(str(output_dir / out_name), diff)

        if config.save_thresholded and thr_dir is not None:
            _, thr = cv2.threshold(diff, config.diff_threshold, 255, cv2.THRESH_BINARY)
            thr_name = f"diff_thr_{curr_path.stem}{config.output_ext}"
            cv2.imwrite(str(thr_dir / thr_name), thr)

        processed += 1
        if config.progress_log_every > 0 and processed % config.progress_log_every == 0:
            print(f"Processed {processed} diffs (frame {idx + 1} / {end_idx + 1})")

    print(f"Done. Wrote {processed} diff images to: {output_dir}")


if __name__ == "__main__":
    process_frames(CONFIG)
