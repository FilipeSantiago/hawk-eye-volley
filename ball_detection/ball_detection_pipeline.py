from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar
    def tqdm(iterable, total=None, desc=None):
        return iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ball_detection.motion_detector import MotionCenterConfig, MotionDetector
from ball_detection.motion_detector.diff_frames import DiffConfig, process_frames
from ball_detection.training.annotated_candidate_dataset_config import (
    AnnotatedCandidateDatasetConfig,
)
from ball_detection.training.candidate_crop_preprocessor import CandidateCropPreprocessor
from ball_detection.training.train_classifier import BallCandidateCNN
from ball_detection.utils.candidate_dataset_constants import IMAGE_EXTENSIONS, PADDING_MODE_TO_CV2


DEFAULT_PROCESSING_ROOT = Path(
    "/home/bugslayer/Downloads/volley video footage/data/processing_folder"
)
DEFAULT_MODEL_PATH = Path("/home/bugslayer/Downloads/volley video footage/models/ball_cnn_best.pt")

# Edit this dict to configure the pipeline.
config = {
    "video": "/home/bugslayer/Downloads/volley video footage/data/videoplayback.1773931843922.publer.com-00.00.00.000-00.10.03.976-00.06.08.875-00.10.04.033.mp4",
    "processing_root": "/home/bugslayer/Downloads/volley video footage/data/processing_folder",
    "model": "/home/bugslayer/Downloads/volley video footage/models/ball_cnn_best.pt",
    "output_video": "",
    "threshold": "0.8",
    "device": "",
    "batch_size": 64,
    "min_moving_area": 30,
    "max_moving_area": 350,
    "diff_threshold": 12,
    "motion_center_strategy": "midpoint",
    "motion_center_component_source": "union",
    "motion_center_diff_threshold": None,
    "motion_center_min_area": 12,
    "motion_center_bbox_size": None,
    "motion_center_bbox_scale": 0.9,
    "motion_center_bbox_min": 8,
    "motion_center_bbox_max": 80,
    "motion_center_appearance_mode": "contrast",
    "motion_center_appearance_blur": [3, 3],
    "motion_center_appearance_gamma": 1.0,
    "motion_center_motion_weight_scale": 0.5,
    "motion_center_appearance_blend": 0.7,
    "motion_center_appearance_padding": 4,
    "expansion_factor": 2.0,
    "output_size": [64, 64],
    "padding_mode": "constant",
    "normalize": True,
    "draw_score": True,
    "draw_frame_index_if_no_ball": True,
    "save_ball_crops": True,
    "ball_crop_padding": 0,
    "codec": "mp4v",
    "image_ext": ".jpg",
    "export_format": "lazy",
    "tracklet_enabled": True,
    "tracklet_max_frame_gap": 1,
    "tracklet_max_center_distance": 80.0,
    "tracklet_min_iou": 0.0,
    "tracklet_max_area_ratio_change": 2.5,
    "tracklet_min_link_score": 0.30,
    "tracklet_min_score_margin": 0.08,
    "tracklet_use_velocity": True,
}


def video_to_frames(
    video_path: str | Path,
    output_path: str | Path,
    image_ext: str = ".jpg",
) -> tuple[int, Path]:
    video_file = Path(video_path).expanduser()
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")
    if not video_file.is_file():
        raise ValueError(f"Video path is not a file: {video_file}")

    if not image_ext.startswith("."):
        image_ext = f".{image_ext}"

    output_dir = Path(output_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / video_file.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_file}")

    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            frame_file = frames_dir / f"frame_{frame_count:06d}{image_ext}"
            saved = cv2.imwrite(str(frame_file), frame)
            if not saved:
                raise RuntimeError(f"Failed to write frame: {frame_file}")
    finally:
        capture.release()

    return frame_count, frames_dir


def iter_image_paths(root_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in root_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def get_video_fps(video_path: Path, fallback: float = 30.0) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 1e-3:
        return fallback
    return fps


def ensure_diff_frames(
    frames_dir: Path,
    diff_dir: Path,
    diff_threshold: int,
    output_ext: str = ".png",
) -> None:
    diff_dir.mkdir(parents=True, exist_ok=True)
    config = DiffConfig(
        input_frames_dir=str(frames_dir),
        output_diff_dir=str(diff_dir),
        start_frame=2,
        end_frame=None,
        use_roi=False,
        save_thresholded=False,
        diff_threshold=diff_threshold,
        normalize_diff=False,
        blur_ksize=(3, 3),
        output_ext=output_ext,
        progress_log_every=200,
    )
    process_frames(config)


def load_model(
    model_path: Path, device: torch.device
) -> Tuple[BallCandidateCNN, Dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    metadata: Dict[str, Any] = {}

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata["threshold"] = checkpoint.get("threshold")
        metadata["in_channels"] = checkpoint.get("in_channels")
        metadata["input_mode"] = checkpoint.get("input_mode")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {model_path}")

    in_channels = int(metadata.get("in_channels") or 4)
    model = BallCandidateCNN(in_channels=in_channels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata


def build_input_tensor(
    sample_image: np.ndarray,
    sample_brightness: np.ndarray,
    normalize: bool,
) -> torch.Tensor:
    rgb = sample_image.astype(np.float32)
    brightness = sample_brightness.astype(np.float32)

    if normalize:
        rgb /= 255.0
        brightness /= 255.0

    brightness = brightness[..., None]
    stacked = np.concatenate([rgb, brightness], axis=2)
    stacked = np.transpose(stacked, (2, 0, 1))
    return torch.from_numpy(stacked)


def crop_bbox_with_padding(
    image: np.ndarray, bbox_xywh: Tuple[int, int, int, int], padding: int
) -> np.ndarray | None:
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        return None
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def annotate_crop_with_prob(
    image: np.ndarray,
    prob: float,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    vis = image.copy()
    label = f"{prob * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    margin = 6
    x = margin
    y = margin + text_size[1]
    box_x1 = max(0, x - margin)
    box_y1 = max(0, y - text_size[1] - margin)
    box_x2 = min(vis.shape[1] - 1, x + text_size[0] + margin)
    box_y2 = min(vis.shape[0] - 1, y + baseline + margin)
    cv2.rectangle(vis, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), thickness=-1)
    cv2.putText(vis, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return vis


def classify_candidates(
    preprocessor: CandidateCropPreprocessor,
    candidates_dir: Path,
    model: BallCandidateCNN,
    device: torch.device,
    batch_size: int,
    threshold: float,
    normalize: bool,
    ball_crops_dir: Path | None = None,
    ball_crop_padding: int = 0,
    ball_crop_ext: str = ".jpg",
) -> Tuple[Dict[str, Dict[str, Any]], List[str], Counter]:
    frame_results: Dict[str, Dict[str, Any]] = {}
    frame_order: List[str] = []
    skipped = Counter()

    batch_inputs: List[torch.Tensor] = []
    batch_meta: List[Dict[str, Any]] = []

    def flush_batch() -> None:
        if not batch_inputs:
            return
        batch = torch.stack(batch_inputs, dim=0).to(device)
        with torch.inference_mode():
            logits = model(batch)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            elif logits.ndim != 1:
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()

        for meta, prob in zip(batch_meta, probs):
            frame_name = meta["frame_name"]
            candidate_index = meta["candidate_index"]
            bbox = meta["bbox"]
            frame_entry = frame_results.setdefault(
                frame_name,
                {
                    "frame": frame_name,
                    "frame_name": frame_name,
                    "index": None,
                    "candidates": [],
                },
            )
            frame_entry["candidates"].append(
                {
                    "candidate_index": candidate_index,
                    "bbox": bbox,
                    "prob": float(prob),
                    "is_ball": bool(prob >= threshold),
                }
            )
            if prob >= threshold:
                save_path = meta.get("ball_crop_path")
                crop_bgr = meta.get("ball_crop_bgr")
                if save_path is not None and crop_bgr is not None:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    annotated = annotate_crop_with_prob(crop_bgr, float(prob))
                    cv2.imwrite(str(save_path), annotated)

        batch_inputs.clear()
        batch_meta.clear()

    candidate_files = sorted(candidates_dir.glob("*.json"))
    for candidate_path in tqdm(
        candidate_files, total=len(candidate_files), desc="Classifying candidates"
    ):
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped["invalid_json"] += 1
            continue

        frame_name = payload.get("frame")
        if not isinstance(frame_name, str) or not frame_name:
            frame_name = f"{candidate_path.stem}.jpg"

        if frame_name not in frame_order:
            frame_order.append(frame_name)

        frame_index = payload.get("index")
        frame_entry = frame_results.setdefault(
            frame_name,
            {
                "frame": frame_name,
                "frame_name": frame_name,
                "index": frame_index,
                "candidates": [],
            },
        )
        if frame_entry.get("index") is None and frame_index is not None:
            frame_entry["index"] = frame_index

        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            skipped["invalid_candidates"] += 1
            continue

        frame_stem = Path(frame_name).stem
        frame_path = preprocessor.find_original_frame(frame_stem)
        if frame_path is None:
            skipped["missing_original_frame"] += 1
            continue
        diff_path = preprocessor.find_diff_frame(frame_stem)
        if diff_path is None:
            skipped["missing_diff_frame"] += 1
            continue

        frame_entry["frame"] = str(frame_path.absolute())
        frame_entry["frame_name"] = frame_name
        frame_entry["diff_frame"] = str(diff_path.absolute())

        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            skipped["failed_to_read_original_frame"] += 1
            continue
        diff_raw = cv2.imread(str(diff_path), cv2.IMREAD_UNCHANGED)
        if diff_raw is None:
            skipped["failed_to_read_diff_image"] += 1
            continue

        if diff_raw.ndim == 2:
            diff_gray = diff_raw
        elif diff_raw.ndim == 3:
            diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        else:
            skipped["invalid_diff_image_shape"] += 1
            continue

        should_save_ball_crops = ball_crops_dir is not None and len(candidates) > 1
        for idx, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                skipped["invalid_candidate_entry"] += 1
                continue
            bbox = candidate.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                skipped["invalid_bbox"] += 1
                continue

            try:
                bbox_xywh = tuple(int(round(float(v))) for v in bbox)
            except (TypeError, ValueError):
                skipped["invalid_bbox"] += 1
                continue
            if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
                skipped["invalid_bbox"] += 1
                continue

            square_crop_xywh = preprocessor.compute_square_crop_region(bbox_xywh)
            if square_crop_xywh is None:
                skipped["invalid_square_crop_region"] += 1
                continue

            rgb_crop_bgr = preprocessor.crop_with_padding(frame_bgr, square_crop_xywh)
            if rgb_crop_bgr is None:
                skipped["failed_rgb_crop_with_padding"] += 1
                continue

            diff_crop = preprocessor.crop_with_padding(diff_gray, square_crop_xywh)
            if diff_crop is None:
                skipped["failed_diff_crop_with_padding"] += 1
                continue

            rgb_crop_bgr, diff_crop = preprocessor.resize_pair(rgb_crop_bgr, diff_crop)
            rgb_crop = cv2.cvtColor(rgb_crop_bgr, cv2.COLOR_BGR2RGB)

            tensor = build_input_tensor(
                sample_image=rgb_crop,
                sample_brightness=diff_crop,
                normalize=normalize,
            )
            batch_inputs.append(tensor)
            meta: Dict[str, Any] = {
                "frame_name": frame_name,
                "candidate_index": idx,
                "bbox": bbox,
            }
            if should_save_ball_crops:
                frame_crop_dir = ball_crops_dir / frame_stem
                crop_path = frame_crop_dir / f"{frame_stem}_candidate_{idx:03d}{ball_crop_ext}"
                crop_bgr = crop_bbox_with_padding(
                    frame_bgr, bbox_xywh=bbox_xywh, padding=ball_crop_padding
                )
                diff_crop = crop_bbox_with_padding(
                    diff_gray, bbox_xywh=bbox_xywh, padding=ball_crop_padding
                )
                if crop_bgr is not None and diff_crop is not None:
                    if diff_crop.ndim == 2:
                        diff_bgr = cv2.cvtColor(diff_crop, cv2.COLOR_GRAY2BGR)
                    else:
                        diff_bgr = diff_crop
                    if diff_bgr.shape[:2] != crop_bgr.shape[:2]:
                        diff_bgr = cv2.resize(
                            diff_bgr,
                            (crop_bgr.shape[1], crop_bgr.shape[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    combined = cv2.hconcat([crop_bgr, diff_bgr])
                    meta["ball_crop_path"] = crop_path
                    meta["ball_crop_bgr"] = combined
            batch_meta.append(meta)

            if len(batch_inputs) >= batch_size:
                flush_batch()

    flush_batch()

    for frame_name, frame_entry in frame_results.items():
        frame_entry["candidates"].sort(key=lambda item: item["candidate_index"])
        frame_entry["num_candidates"] = len(frame_entry["candidates"])

    return frame_results, frame_order, skipped


def infer_frame_index(frame_name: str | None) -> int | None:
    if not isinstance(frame_name, str) or not frame_name:
        return None
    stem = Path(frame_name).stem
    if stem.startswith("frame_"):
        suffix = stem.split("frame_", 1)[1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return int(digits)
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    if digits:
        return int(digits)
    return None


def bbox_center(bbox_xywh: Iterable[float]) -> Tuple[float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return (x + w * 0.5, y + h * 0.5)


def bbox_area(bbox_xywh: Iterable[float]) -> float:
    _, _, w, h = [float(v) for v in bbox_xywh]
    if w <= 0.0 or h <= 0.0:
        return 0.0
    return w * h


def bbox_iou(bbox_a: Iterable[float], bbox_b: Iterable[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in bbox_a]
    bx, by, bw, bh = [float(v) for v in bbox_b]
    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return 0.0
    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def bbox_area_ratio(bbox_a: Iterable[float], bbox_b: Iterable[float]) -> float:
    area_a = bbox_area(bbox_a)
    area_b = bbox_area(bbox_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return float("inf")
    return max(area_a, area_b) / min(area_a, area_b)


def center_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return float(math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def score_tracklet_link(
    candidate_bbox: Iterable[float],
    tracklet_state: Dict[str, Any],
    frame_gap: int,
    max_frame_gap: int,
    max_center_distance: float,
    min_iou: float,
    max_area_ratio_change: float,
    use_velocity: bool,
) -> float:
    last_bbox = tracklet_state.get("last_bbox")
    if last_bbox is None:
        return 0.0

    iou = bbox_iou(last_bbox, candidate_bbox)
    if iou < float(min_iou):
        return 0.0

    area_ratio = bbox_area_ratio(last_bbox, candidate_bbox)
    if area_ratio > float(max_area_ratio_change):
        return 0.0

    candidate_center = bbox_center(candidate_bbox)
    last_center = tracklet_state.get("last_center")
    if last_center is None:
        return 0.0

    distance = center_distance(last_center, candidate_center)
    if use_velocity:
        velocity = tracklet_state.get("velocity")
        if velocity is not None:
            pred_center = (last_center[0] + velocity[0], last_center[1] + velocity[1])
            pred_distance = center_distance(pred_center, candidate_center)
            distance = min(distance, pred_distance)

    if max_center_distance > 0.0 and distance > max_center_distance:
        return 0.0

    if max_center_distance <= 0.0:
        distance_score = 1.0 if distance <= 1e-6 else 0.0
    else:
        distance_score = max(0.0, 1.0 - distance / max_center_distance)

    iou_score = float(iou)

    if max_area_ratio_change <= 1.0:
        area_score = 1.0 if area_ratio <= max_area_ratio_change else 0.0
    else:
        area_score = 1.0 - ((area_ratio - 1.0) / (max_area_ratio_change - 1.0))
        area_score = max(0.0, min(1.0, area_score))

    gap_penalty = 1.0
    if max_frame_gap > 0 and frame_gap > 0:
        gap_penalty = max(0.0, 1.0 - frame_gap / (max_frame_gap + 1))

    score = (distance_score + iou_score + area_score) / 3.0
    return score * gap_penalty


def build_frame_sequence(
    frame_results: Dict[str, Dict[str, Any]], frame_order: List[str]
) -> List[Tuple[str, Dict[str, Any], int]]:
    sequence: List[Tuple[str, Dict[str, Any], int]] = []
    for position, frame_name in enumerate(frame_order, start=1):
        frame_entry = frame_results.get(frame_name)
        if not frame_entry:
            continue
        frame_index = frame_entry.get("index")
        if frame_index is None:
            inferred = infer_frame_index(frame_entry.get("frame") or frame_name)
            frame_index = inferred if inferred is not None else position
            frame_entry["index"] = frame_index
        sequence.append((frame_name, frame_entry, int(frame_index)))
    return sequence


def append_tracklet_sample(
    tracklet: Dict[str, Any],
    tracklet_state: Dict[str, Any],
    candidate: Dict[str, Any],
    frame_name: str,
    frame_index: int,
) -> None:
    bbox = candidate.get("bbox")
    if not bbox or len(bbox) != 4:
        return
    center = bbox_center(bbox)
    sample = {
        "frame": frame_name,
        "index": frame_index,
        "candidate_index": candidate.get("candidate_index"),
        "bbox": bbox,
        "prob": float(candidate.get("prob", 0.0)),
        "is_ball": bool(candidate.get("is_ball")),
    }
    tracklet.setdefault("samples", []).append(sample)
    tracklet["end_frame"] = frame_name
    tracklet["end_index"] = frame_index
    tracklet["num_samples"] = len(tracklet["samples"])
    candidate["tracklet_id"] = tracklet["tracklet_id"]

    prev_center = tracklet_state.get("last_center")
    tracklet_state["prev_center"] = prev_center
    tracklet_state["last_center"] = center
    tracklet_state["last_bbox"] = bbox
    tracklet_state["last_frame_index"] = frame_index
    if prev_center is not None:
        tracklet_state["velocity"] = (
            center[0] - prev_center[0],
            center[1] - prev_center[1],
        )


def build_tracklets(
    frame_results: Dict[str, Dict[str, Any]],
    frame_order: List[str],
    max_frame_gap: int,
    max_center_distance: float,
    min_iou: float,
    max_area_ratio_change: float,
    min_link_score: float,
    min_score_margin: float,
    use_velocity: bool,
) -> List[Dict[str, Any]]:
    tracklets: Dict[int, Dict[str, Any]] = {}
    active: Dict[int, Dict[str, Any]] = {}
    next_tracklet_id = 1

    for frame_name, frame_entry, frame_index in build_frame_sequence(
        frame_results, frame_order
    ):
        frame_path = frame_entry.get("frame") or frame_name
        candidates = frame_entry.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []

        for tracklet_id in list(active.keys()):
            last_index = active[tracklet_id].get("last_frame_index")
            if last_index is None:
                continue
            if frame_index - int(last_index) > max_frame_gap:
                active.pop(tracklet_id, None)

        candidate_links: List[Tuple[float, Dict[str, Any], int]] = []
        for candidate in candidates:
            bbox = candidate.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            scores: List[Tuple[float, int]] = []
            for tracklet_id, state in active.items():
                last_index = state.get("last_frame_index")
                if last_index is None:
                    continue
                frame_gap = frame_index - int(last_index)
                if frame_gap <= 0:
                    frame_gap = 0
                if frame_gap > max_frame_gap:
                    continue
                score = score_tracklet_link(
                    candidate_bbox=bbox,
                    tracklet_state=state,
                    frame_gap=frame_gap,
                    max_frame_gap=max_frame_gap,
                    max_center_distance=max_center_distance,
                    min_iou=min_iou,
                    max_area_ratio_change=max_area_ratio_change,
                    use_velocity=use_velocity,
                )
                if score > 0.0:
                    scores.append((score, tracklet_id))
            if not scores:
                continue
            scores.sort(key=lambda item: item[0], reverse=True)
            best_score, best_tracklet = scores[0]
            second_score = scores[1][0] if len(scores) > 1 else 0.0
            if best_score < min_link_score:
                continue
            if (best_score - second_score) < min_score_margin:
                continue
            candidate_links.append((best_score, candidate, best_tracklet))

        candidate_links.sort(key=lambda item: item[0], reverse=True)
        used_tracklets: set[int] = set()
        for _, candidate, tracklet_id in candidate_links:
            if tracklet_id in used_tracklets:
                continue
            if candidate.get("tracklet_id") is not None:
                continue
            tracklet = tracklets.get(tracklet_id)
            state = active.get(tracklet_id)
            if tracklet is None or state is None:
                continue
            append_tracklet_sample(
                tracklet=tracklet,
                tracklet_state=state,
                candidate=candidate,
                frame_name=frame_path,
                frame_index=frame_index,
            )
            used_tracklets.add(tracklet_id)

        for candidate in candidates:
            bbox = candidate.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            if candidate.get("tracklet_id") is not None:
                continue
            tracklet_id = next_tracklet_id
            next_tracklet_id += 1
            tracklet = {
                "tracklet_id": tracklet_id,
                "start_frame": frame_path,
                "end_frame": frame_path,
                "start_index": frame_index,
                "end_index": frame_index,
                "samples": [],
                "num_samples": 0,
            }
            tracklets[tracklet_id] = tracklet
            state = {
                "tracklet_id": tracklet_id,
                "last_frame_index": frame_index,
                "last_bbox": None,
                "last_center": None,
                "prev_center": None,
                "velocity": None,
            }
            active[tracklet_id] = state
            append_tracklet_sample(
                tracklet=tracklet,
                tracklet_state=state,
                candidate=candidate,
                frame_name=frame_path,
                frame_index=frame_index,
            )

    return [tracklets[tracklet_id] for tracklet_id in sorted(tracklets)]


def build_frame_views(frame_entry: Dict[str, Any]) -> Dict[str, Any]:
    candidates = frame_entry.get("candidates") or []
    if not candidates:
        return {
            "most_likely_ball_object": None,
            "classified_ball_objects": [],
            "other_moving_objects": [],
            "best_candidate": None,
            "ball_candidates": [],
            "other_candidates": [],
        }
    best_candidate = max(
        candidates, key=lambda item: float(item.get("prob", -1.0))
    )
    ball_candidates = [candidate for candidate in candidates if candidate.get("is_ball")]
    other_candidates = [
        candidate for candidate in candidates if not candidate.get("is_ball")
    ]
    return {
        "most_likely_ball_object": best_candidate,
        "classified_ball_objects": ball_candidates,
        "other_moving_objects": other_candidates,
        "best_candidate": best_candidate,
        "ball_candidates": ball_candidates,
        "other_candidates": other_candidates,
    }


def enrich_frame_candidates(frame_entry: Dict[str, Any]) -> None:
    frame_name = frame_entry.get("frame_name")
    if not isinstance(frame_name, str) or not frame_name:
        frame_value = frame_entry.get("frame")
        frame_name = Path(frame_value).name if isinstance(frame_value, str) else ""
        frame_entry["frame_name"] = frame_name

    frame_index = frame_entry.get("index")
    candidates = frame_entry.get("candidates") or []
    if not isinstance(candidates, list):
        return

    for idx, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        if candidate.get("candidate_index") is None:
            candidate["candidate_index"] = idx
        candidate_index = candidate.get("candidate_index")
        try:
            candidate_index_value = int(candidate_index)
        except (TypeError, ValueError):
            candidate_index_value = idx
            candidate["candidate_index"] = candidate_index_value

        candidate["frame_name"] = frame_name
        candidate["frame_index"] = frame_index

        candidate["proposal_tracklet_id"] = candidate.get("tracklet_id")

        candidate_uid = candidate.get("candidate_uid")
        if not isinstance(candidate_uid, str) or not candidate_uid.strip():
            if isinstance(frame_index, int):
                frame_token = f"{frame_index:06d}"
            else:
                frame_token = Path(frame_name).stem if frame_name else "frame"
            candidate_uid = f"{frame_token}_candidate_{candidate_index_value:03d}"
            candidate["candidate_uid"] = candidate_uid


def validate_frame_record(frame_entry: Dict[str, Any]) -> None:
    if "frame" not in frame_entry:
        raise ValueError("Frame record missing 'frame' field.")
    frame_value = frame_entry.get("frame")
    if not isinstance(frame_value, str) or not frame_value.strip():
        raise ValueError("Frame record has invalid 'frame' value.")
    if not Path(frame_value).is_absolute():
        raise ValueError("Frame record 'frame' must be an absolute path.")
    diff_value = frame_entry.get("diff_frame")
    if isinstance(diff_value, str) and diff_value:
        if not Path(diff_value).is_absolute():
            raise ValueError("Frame record 'diff_frame' must be an absolute path.")
    if "index" not in frame_entry or frame_entry.get("index") is None:
        raise ValueError("Frame record missing 'index' field.")
    candidates = frame_entry.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Frame record missing 'candidates' array.")

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Candidate entry must be a dict.")
        candidate_uid = candidate.get("candidate_uid")
        if not isinstance(candidate_uid, str) or not candidate_uid.strip():
            raise ValueError("Candidate missing non-empty 'candidate_uid'.")
        if "proposal_tracklet_id" not in candidate:
            raise ValueError("Candidate missing 'proposal_tracklet_id'.")
        if candidate.get("candidate_index") is None:
            raise ValueError("Candidate missing 'candidate_index'.")
        if candidate.get("bbox") is None and candidate.get("bbox_xywh") is None:
            raise ValueError("Candidate missing 'bbox'/'bbox_xywh'.")
        if candidate.get("prob") is None:
            raise ValueError("Candidate missing 'prob'.")
        frame_name = candidate.get("frame_name")
        if not isinstance(frame_name, str) or not frame_name.strip():
            raise ValueError("Candidate missing non-empty 'frame_name'.")
        if "frame_index" not in candidate:
            raise ValueError("Candidate missing 'frame_index'.")


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding=encoding,
            delete=False,
            dir=path.parent,
            prefix=path.name,
            suffix=".tmp",
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            tmp_file.write(text)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists() and temp_path != path:
            try:
                temp_path.unlink()
            except OSError:
                pass


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2), encoding="utf-8")


def write_frames_ndjson(
    frame_order: List[str],
    frame_results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_total = sum(1 for name in frame_order if name in frame_results)

    index_items: List[Dict[str, Any]] = []
    counts = {
        "total_frames": 0,
        "total_candidates": 0,
        "total_ball_candidates": 0,
        "frames_with_candidates": 0,
        "frames_with_ball_candidates": 0,
    }

    temp_path: Path | None = None
    offset = 0
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            delete=False,
            dir=output_path.parent,
            prefix=output_path.name,
            suffix=".tmp",
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            for frame_name in frame_order:
                frame_entry = frame_results.get(frame_name)
                if not frame_entry:
                    continue
                validate_frame_record(frame_entry)
                candidates = frame_entry.get("candidates") or []
                counts["total_candidates"] += len(candidates)
                ball_candidates = [candidate for candidate in candidates if candidate.get("is_ball")]
                counts["total_ball_candidates"] += len(ball_candidates)
                if candidates:
                    counts["frames_with_candidates"] += 1
                if ball_candidates:
                    counts["frames_with_ball_candidates"] += 1

                line = json.dumps(frame_entry, separators=(",", ":"))
                line_bytes = line.encode("utf-8") + b"\n"
                tmp_file.write(line_bytes)

                frame_name_value = frame_entry.get("frame_name")
                if not frame_name_value:
                    frame_value = frame_entry.get("frame")
                    frame_name_value = (
                        Path(frame_value).name if isinstance(frame_value, str) else ""
                    )
                frame_index_value = frame_entry.get("index")
                try:
                    frame_index_value = int(frame_index_value)
                except (TypeError, ValueError):
                    pass

                index_items.append(
                    {
                        "frame_index": frame_index_value,
                        "frame_name": frame_name_value,
                        "offset": offset,
                        "length": len(line_bytes),
                    }
                )
                offset += len(line_bytes)
                counts["total_frames"] += 1

            tmp_file.flush()
            os.fsync(tmp_file.fileno())
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise

    if counts["total_frames"] != expected_total:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise ValueError(
            f"frames.ndjson line count {counts['total_frames']} "
            f"does not match expected {expected_total}."
        )

    if temp_path is None:
        raise RuntimeError("Failed to write frames.ndjson.")
    try:
        os.replace(temp_path, output_path)
    except OSError:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return index_items, counts


def draw_candidate_boxes(
    image: np.ndarray,
    candidates: Iterable[Dict[str, Any]],
    highlight_color: Tuple[int, int, int] = (0, 255, 0),
    other_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    draw_score: bool = True,
) -> np.ndarray:
    vis = image.copy()
    img_h, img_w = vis.shape[:2]
    candidates_list = list(candidates)
    if not candidates_list:
        return vis

    best_candidate = max(candidates_list, key=lambda item: float(item.get("prob", -1.0)))
    best_index = best_candidate.get("candidate_index")

    for candidate in candidates_list:
        bbox = candidate.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = [int(round(v)) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        x1 = max(0, min(img_w - 1, x))
        y1 = max(0, min(img_h - 1, y))
        x2 = max(0, min(img_w - 1, x + w))
        y2 = max(0, min(img_h - 1, y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        color = highlight_color if candidate.get("candidate_index") == best_index else other_color
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if draw_score:
            prob = float(candidate.get("prob", 0.0))
            label = f"{prob:.2f}"
            cv2.putText(
                vis,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return vis


def write_annotated_video(
    frames_dir: Path,
    predictions_by_frame: Dict[str, List[Dict[str, Any]]],
    frame_meta_by_stem: Dict[str, Dict[str, Any]],
    output_path: Path,
    fps: float,
    codec: str = "mp4v",
    draw_score: bool = True,
    draw_frame_index_if_no_ball: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        for frame_path in iter_image_paths(frames_dir):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            candidates = predictions_by_frame.get(frame_path.stem, [])
            vis = draw_candidate_boxes(frame, candidates, draw_score=draw_score)

            if draw_frame_index_if_no_ball:
                meta = frame_meta_by_stem.get(frame_path.stem)
                if meta and meta.get("num_candidates", 0) > 0 and meta.get("ball_count", 0) == 0:
                    frame_index = meta.get("index")
                    if frame_index is None:
                        stem = frame_path.stem
                        if stem.startswith("frame_"):
                            try:
                                frame_index = int(stem.split("_", 1)[1])
                            except ValueError:
                                frame_index = None
                    label = f"FRAME {frame_index}" if frame_index is not None else frame_path.stem
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.3
                    thickness = 4
                    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    margin = 14
                    x = max(margin, vis.shape[1] - text_size[0] - margin)
                    y = margin + text_size[1]
                    # Draw a solid black box behind the text for visibility.
                    box_pad = 5
                    box_x1 = max(0, x - box_pad)
                    box_y1 = max(0, y - text_size[1] - box_pad)
                    box_x2 = min(vis.shape[1] - 1, x + text_size[0] + box_pad)
                    box_y2 = min(vis.shape[0] - 1, y + baseline + box_pad)
                    cv2.rectangle(vis, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), thickness=-1)
                    # Draw thick yellow text on top.
                    cv2.putText(
                        vis,
                        label,
                        (x, y),
                        font,
                        font_scale,
                        (0, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

            if writer is None:
                height, width = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    str(output_path), fourcc, fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {output_path}")
            writer.write(vis)
    finally:
        if writer is not None:
            writer.release()


def build_processing_paths(processing_root: Path, video_stem: str) -> Dict[str, Path]:
    return {
        "frames_dir": processing_root / "frames" / video_stem,
        "diff_dir": processing_root / "diff_frames" / video_stem,
        "motion_annotated_dir": processing_root / "motion_annotated" / video_stem,
        "motion_candidates_dir": processing_root / "motion_candidates" / video_stem,
        "candidate_crops_dir": processing_root / "motion_candidate_crops" / video_stem,
        "ball_candidate_crops_dir": processing_root / "ball_candidate_crops" / video_stem,
        "predictions_dir": processing_root / "ball_predictions",
        "annotations_dir": processing_root / "ball_annotations",
    }


def build_preprocessor(
    frames_dir: Path,
    candidates_dir: Path,
    diff_dir: Path,
    expansion_factor: float,
    output_size: Tuple[int, int],
    padding_mode: str,
) -> CandidateCropPreprocessor:
    config = AnnotatedCandidateDatasetConfig(
        annotated_ball_dir=frames_dir,
        annotated_not_ball_dir=frames_dir,
        output_ball_dir=frames_dir,
        output_not_ball_dir=frames_dir,
        frames_dir=frames_dir,
        candidates_dir=candidates_dir,
        diff_frames_dir=diff_dir,
        expansion_factor=expansion_factor,
        output_size=output_size,
        padding_mode=padding_mode,
        preview_enabled=False,
        preview_ext=".jpg",
        strict_filename_parsing=True,
        flatten_output=True,
    )
    return CandidateCropPreprocessor(config)


def normalize_config(raw_config: Dict[str, Any] = None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "video": "/home/bugslayer/Downloads/volley video footage/data/YTDown.com_YouTube_BARBARAS-VOLEI-X-ARARIBOIA-VOLEI-LIGA-B-_Media_32eiEkxLjfE_001_720p-00.15.55.124-00.24.27.801.mp4",
        "processing_root": str(DEFAULT_PROCESSING_ROOT),
        "model": str(DEFAULT_MODEL_PATH),
        "output_video": "",
        "threshold": "",
        "device": "",
        "batch_size": 128,
        "min_moving_area": 5,
        "max_moving_area": 750,
        "diff_threshold": 12,
        "expansion_factor": 2.0,
        "output_size": [64, 64],
        "padding_mode": "constant",
        "normalize": True,
        "draw_score": True,
        "draw_frame_index_if_no_ball": True,
        "save_ball_crops": True,
        "ball_crop_padding": 0,
        "codec": "mp4v",
        "image_ext": ".jpg",
        "export_format": "lazy",
        "tracklet_enabled": True,
        "tracklet_max_frame_gap": 1,
        "tracklet_max_center_distance": 80.0,
        "tracklet_min_iou": 0.0,
        "tracklet_max_area_ratio_change": 2.5,
        "tracklet_min_link_score": 0.30,
        "tracklet_min_score_margin": 0.08,
        "tracklet_use_velocity": True,
    }
    if not raw_config:
        return merged

    merged.update(raw_config or {})

    if not merged.get("video"):
        raise ValueError("Config must include 'video' path.")
    if not isinstance(merged.get("output_size"), (list, tuple)) or len(merged["output_size"]) != 2:
        raise ValueError("Config 'output_size' must be a list/tuple of two ints.")

    return merged


def run_pipeline(raw_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_version = getattr(torch.version, "cuda", None)
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"CUDA version: {cuda_version or 'unknown'}")
    print(f"GPU: {gpu_name or 'unknown'}")

    config = normalize_config(raw_config) if raw_config is not None else normalize_config()

    video_path = Path(config["video"]).expanduser().absolute()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    processing_root = Path(config["processing_root"]).expanduser().absolute()
    processing_root.mkdir(parents=True, exist_ok=True)
    video_stem = video_path.stem
    paths = build_processing_paths(processing_root, video_stem)

    frames_root = paths["frames_dir"].parent
    saved_frames, frames_dir = video_to_frames(
        video_path, frames_root, image_ext=config["image_ext"]
    )
    frames_dir = frames_dir.absolute()
    print(f"Extracted {saved_frames} frame(s) to {frames_dir}")

    diff_dir = paths["diff_dir"]
    diff_dir = diff_dir.absolute()
    print("Computing diff frames...")
    ensure_diff_frames(
        frames_dir=frames_dir,
        diff_dir=diff_dir,
        diff_threshold=int(config["diff_threshold"]),
    )

    bbox_size_value = config.get("motion_center_bbox_size")
    if bbox_size_value in (None, ""):
        bbox_size = None
    else:
        bbox_size = int(bbox_size_value)

    blur_value = config.get("motion_center_appearance_blur")
    if isinstance(blur_value, (list, tuple)) and len(blur_value) == 2:
        appearance_blur = (int(blur_value[0]), int(blur_value[1]))
    else:
        appearance_blur = (3, 3)

    diff_override = config.get("motion_center_diff_threshold")
    if diff_override in (None, ""):
        diff_override_value = None
    else:
        diff_override_value = int(diff_override)

    center_config = MotionCenterConfig(
        strategy=str(config.get("motion_center_strategy") or "midpoint"),
        component_source=str(config.get("motion_center_component_source") or "union"),
        diff_threshold=diff_override_value,
        min_center_area=int(config.get("motion_center_min_area") or 12),
        bbox_size=bbox_size,
        bbox_scale=float(config.get("motion_center_bbox_scale") or 0.9),
        bbox_min_size=int(config.get("motion_center_bbox_min") or 8),
        bbox_max_size=int(config.get("motion_center_bbox_max") or 80),
        appearance_weight_mode=str(config.get("motion_center_appearance_mode") or "contrast"),
        appearance_blur_ksize=appearance_blur,
        appearance_gamma=float(config.get("motion_center_appearance_gamma") or 1.0),
        motion_weight_scale=float(config.get("motion_center_motion_weight_scale") or 0.5),
        appearance_blend=float(config.get("motion_center_appearance_blend") or 0.7),
        appearance_padding=int(config.get("motion_center_appearance_padding") or 4),
    )

    motion_detector = MotionDetector(
        frames_dir=frames_dir,
        annotated_dir=paths["motion_annotated_dir"],
        candidates_dir=paths["motion_candidates_dir"],
        min_moving_area=int(config["min_moving_area"]),
        max_moving_area=int(config["max_moving_area"]),
        diff_threshold=int(config["diff_threshold"]),
        center_config=center_config,
    )
    processed = motion_detector.run_motion_detector()
    print(f"Motion detector processed {processed} frame pairs.")

    exported = motion_detector.export_cropped_candidates(paths["candidate_crops_dir"])
    print(f"Exported {exported} candidate crops to {paths['candidate_crops_dir']}")

    device_value = str(config.get("device") or "").strip()
    if device_value:
        device = torch.device(device_value)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(config["model"]).expanduser().absolute()
    model, metadata = load_model(model_path, device=device)
    threshold_value = config.get("threshold")
    if threshold_value == "" or threshold_value is None:
        threshold = float(metadata.get("threshold") or 0.5)
    else:
        threshold = float(threshold_value)
    print(f"Using device: {device}")
    print(f"Using threshold: {threshold:.3f}")

    preprocessor = build_preprocessor(
        frames_dir=frames_dir,
        candidates_dir=paths["motion_candidates_dir"],
        diff_dir=diff_dir,
        expansion_factor=float(config["expansion_factor"]),
        output_size=(int(config["output_size"][0]), int(config["output_size"][1])),
        padding_mode=str(config["padding_mode"]),
    )

    print("Classifying candidates...")
    ball_crops_dir = None
    if bool(config.get("save_ball_crops")):
        ball_crops_dir = paths["ball_candidate_crops_dir"]
        ball_crops_dir.mkdir(parents=True, exist_ok=True)

    image_ext = str(config.get("image_ext") or ".jpg")
    if not image_ext.startswith("."):
        image_ext = f".{image_ext}"

    frame_results, frame_order, skipped = classify_candidates(
        preprocessor=preprocessor,
        candidates_dir=paths["motion_candidates_dir"],
        model=model,
        device=device,
        batch_size=max(1, int(config["batch_size"])),
        threshold=threshold,
        normalize=bool(config["normalize"]),
        ball_crops_dir=ball_crops_dir,
        ball_crop_padding=int(config.get("ball_crop_padding", 0)),
        ball_crop_ext=image_ext,
    )

    for frame_name, frame_entry in frame_results.items():
        frame_value = frame_entry.get("frame") or frame_name
        frame_path = Path(frame_value)
        if not frame_path.is_absolute():
            frame_path = frames_dir / frame_path.name
        frame_entry["frame"] = str(frame_path.absolute())
        frame_entry.setdefault("frame_name", frame_name)
        diff_value = frame_entry.get("diff_frame")
        if isinstance(diff_value, str) and diff_value:
            diff_path = Path(diff_value)
            if not diff_path.is_absolute():
                diff_path = diff_dir / diff_path.name
                frame_entry["diff_frame"] = str(diff_path.absolute())

    tracklet_config = {
        "enabled": bool(config.get("tracklet_enabled", True)),
        "max_frame_gap": int(config.get("tracklet_max_frame_gap", 1)),
        "max_center_distance": float(config.get("tracklet_max_center_distance", 80.0)),
        "min_iou": float(config.get("tracklet_min_iou", 0.0)),
        "max_area_ratio_change": float(
            config.get("tracklet_max_area_ratio_change", 2.5)
        ),
        "min_link_score": float(config.get("tracklet_min_link_score", 0.30)),
        "min_score_margin": float(config.get("tracklet_min_score_margin", 0.08)),
        "use_velocity": bool(config.get("tracklet_use_velocity", True)),
    }

    tracklets: List[Dict[str, Any]] = []
    if tracklet_config["enabled"]:
        tracklets = build_tracklets(
            frame_results=frame_results,
            frame_order=frame_order,
            max_frame_gap=tracklet_config["max_frame_gap"],
            max_center_distance=tracklet_config["max_center_distance"],
            min_iou=tracklet_config["min_iou"],
            max_area_ratio_change=tracklet_config["max_area_ratio_change"],
            min_link_score=tracklet_config["min_link_score"],
            min_score_margin=tracklet_config["min_score_margin"],
            use_velocity=tracklet_config["use_velocity"],
        )
    else:
        build_frame_sequence(frame_results, frame_order)

    for frame_name in frame_order:
        frame_entry = frame_results.get(frame_name)
        if not frame_entry:
            continue
        views = build_frame_views(frame_entry)
        frame_entry["views"] = views
        frame_entry["most_likely_ball_object"] = views.get("most_likely_ball_object")
        frame_entry["classified_ball_objects"] = views.get("classified_ball_objects", [])
        frame_entry["other_moving_objects"] = views.get("other_moving_objects", [])
        enrich_frame_candidates(frame_entry)

    predictions_dir = paths["predictions_dir"]
    predictions_dir.mkdir(parents=True, exist_ok=True)

    export_format = str(config.get("export_format") or "lazy").strip().lower()
    if export_format not in {"monolithic", "lazy"}:
        raise ValueError("Config 'export_format' must be 'monolithic' or 'lazy'.")

    export_payload: Dict[str, Any]
    if export_format == "monolithic":
        predictions_path = predictions_dir / f"{video_stem}.json"
        frames_payload = [
            frame_results[name] for name in frame_order if name in frame_results
        ]
        predictions_payload = {
            "video": str(video_path),
            "frames_dir": str(frames_dir),
            "diff_frames_dir": str(diff_dir),
            "candidates_dir": str(paths["motion_candidates_dir"].absolute()),
            "ball_candidate_crops_dir": str(ball_crops_dir.absolute()) if ball_crops_dir else "",
            "model": str(model_path),
            "threshold": threshold,
            "tracklet_config": tracklet_config,
            "tracklets": tracklets,
            "frames": frames_payload,
            "skipped": dict(skipped),
        }
        write_json_atomic(predictions_path, predictions_payload)
        print(f"Wrote predictions to {predictions_path}")
        export_payload = predictions_payload
    else:
        session_dir = predictions_dir / video_stem
        session_dir.mkdir(parents=True, exist_ok=True)
        frames_path = session_dir / "frames.ndjson"
        index_path = session_dir / "frames.index.json"
        meta_path = session_dir / "session.meta.json"

        index_items, counts = write_frames_ndjson(
            frame_order=frame_order,
            frame_results=frame_results,
            output_path=frames_path,
        )
        index_payload = {
            "format": "ndjson-index-v1",
            "count": counts["total_frames"],
            "items": index_items,
        }
        write_json_atomic(index_path, index_payload)

        summary = {
            "frames_with_candidates": counts["frames_with_candidates"],
            "frames_with_ball_candidates": counts["frames_with_ball_candidates"],
            "total_candidates": counts["total_candidates"],
            "total_ball_candidates": counts["total_ball_candidates"],
            "total_tracklets": len(tracklets),
        }
        session_meta = {
            "video": str(video_path),
            "frames_dir": str(frames_dir),
            "diff_frames_dir": str(diff_dir),
            "candidates_dir": str(paths["motion_candidates_dir"].absolute()),
            "ball_candidate_crops_dir": str(ball_crops_dir.absolute()) if ball_crops_dir else "",
            "model": str(model_path),
            "threshold": threshold,
            "summary": summary,
            "tracklets": tracklets,
            "skipped": dict(skipped),
            "total_frames": counts["total_frames"],
            "export_version": "2",
            "frame_data_format": "ndjson",
        }
        if "frames" in session_meta:
            raise ValueError("session.meta.json must not include 'frames'.")
        write_json_atomic(meta_path, session_meta)

        print(f"Wrote session meta to {meta_path}")
        print(f"Wrote frames to {frames_path}")
        print(f"Wrote frames index to {index_path}")
        export_payload = session_meta

    predictions_by_stem: Dict[str, List[Dict[str, Any]]] = {}
    frame_meta_by_stem: Dict[str, Dict[str, Any]] = {}
    for frame_name in frame_order:
        frame_entry = frame_results.get(frame_name)
        if not frame_entry:
            continue
        stem = Path(frame_entry["frame"]).stem
        candidates = frame_entry["candidates"]
        ball_candidates = [candidate for candidate in candidates if candidate.get("is_ball")]
        predictions_by_stem[stem] = ball_candidates
        frame_meta_by_stem[stem] = {
            "index": frame_entry.get("index"),
            "num_candidates": frame_entry.get("num_candidates", len(candidates)),
            "ball_count": len(ball_candidates),
        }

    output_value = str(config.get("output_video") or "").strip()
    if output_value:
        output_video_path = Path(output_value).expanduser()
    else:
        output_video_path = paths["annotations_dir"] / f"{video_stem}.mp4"
    fps = get_video_fps(video_path)
    print(f"Writing annotated video at {fps:.2f} FPS to {output_video_path}")
    write_annotated_video(
        frames_dir=frames_dir,
        predictions_by_frame=predictions_by_stem,
        frame_meta_by_stem=frame_meta_by_stem,
        output_path=output_video_path,
        fps=fps,
        codec=str(config["codec"]),
        draw_score=bool(config["draw_score"]),
        draw_frame_index_if_no_ball=bool(config.get("draw_frame_index_if_no_ball")),
    )
    print("Done.")
    return export_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the volleyball ball pipeline.")
    parser.add_argument(
        "--export-format",
        choices=["monolithic", "lazy"],
        default="lazy",
        help="Export format for predictions output.",
    )
    args = parser.parse_args()
    run_pipeline({"export_format": args.export_format})


if __name__ == "__main__":
    main()
