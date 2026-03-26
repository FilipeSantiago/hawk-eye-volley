from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from ball_detection.training.annotated_candidate_dataset_config import (
    AnnotatedCandidateDatasetConfig,
)
from ball_detection.training.candidate_crop_preprocessor import CandidateCropPreprocessor
from ball_detection.utils.candidate_dataset_constants import TOKEN_SANITIZER_PATTERN
from ball_detection.utils.dataset_build_summary import DatasetBuildSummary


DEFAULT_ANNOTATIONS_PATH = Path(
    "/home/bugslayer/Downloads/volley video footage/data/processing_folder/"
    "ball_predictions/videoplayback.1773931843922.publer.com-00.00.00.000-00.10.03.976-00.06.08.875-00.10.04.033/"
    "session.meta.annotations.json"
)


def sanitize_token(value: str) -> str:
    sanitized = TOKEN_SANITIZER_PATTERN.sub("_", value.strip())
    return sanitized.strip("_")


def build_preview_image(rgb_image: np.ndarray, brightness: np.ndarray) -> np.ndarray:
    rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    brightness_vis = cv2.cvtColor(brightness, cv2.COLOR_GRAY2BGR)
    if brightness_vis.shape[:2] != rgb_bgr.shape[:2]:
        brightness_vis = cv2.resize(
            brightness_vis,
            (rgb_bgr.shape[1], rgb_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.hconcat([rgb_bgr, brightness_vis])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val datasets from annotation exports."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS_PATH,
        help="Path to session.meta.annotations.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for dataset output (train/val).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (frame-level).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        default=(64, 64),
        metavar=("WIDTH", "HEIGHT"),
        help="Output crop size (width height).",
    )
    parser.add_argument(
        "--expansion-factor",
        type=float,
        default=2.0,
        help="Expansion factor applied to candidate bboxes.",
    )
    parser.add_argument(
        "--padding-mode",
        type=str,
        default="constant",
        help="Padding mode for out-of-bounds crops.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write preview JPEGs alongside npz files.",
    )
    parser.add_argument(
        "--preview-ext",
        type=str,
        default=".jpg",
        help="Preview image extension.",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> tuple[Dict[str, Dict[str, Any]], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions_path = payload.get("predictions_path")
    if not predictions_path:
        raise ValueError("annotations file missing predictions_path")

    annotations: Dict[str, Dict[str, Any]] = {}
    for entry in payload.get("frame_annotations", []) or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("label_source") == "none":
            continue
        frame_name = entry.get("frame_name")
        if not isinstance(frame_name, str) or not frame_name:
            continue
        existing = annotations.get(frame_name)
        if existing is None:
            annotations[frame_name] = entry
            continue
        updated_at = entry.get("updated_at") or ""
        existing_updated = existing.get("updated_at") or ""
        if updated_at >= existing_updated:
            annotations[frame_name] = entry

    return annotations, Path(predictions_path)


def build_split_sets(
    frame_names: list[str], val_ratio: float, seed: int
) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = frame_names[:]
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    val_count = max(0, min(len(shuffled), val_count))
    val_frames = set(shuffled[:val_count])
    train_frames = set(shuffled[val_count:])
    return train_frames, val_frames


def ensure_output_dirs(root: Path, preview_enabled: bool, preview_dir: str, npz_dir: str) -> None:
    (root / npz_dir).mkdir(parents=True, exist_ok=True)
    if preview_enabled:
        (root / preview_dir).mkdir(parents=True, exist_ok=True)


def save_sample(
    output_dir: Path,
    sample_stem: str,
    sample,
    label: int,
    label_name: str,
    expansion_factor: float,
    output_size: tuple[int, int],
    preview_enabled: bool,
    preview_dir: str,
    preview_ext: str,
    npz_dir: str,
) -> None:
    npz_path = (output_dir / npz_dir / sample_stem).with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(npz_path),
        image=sample.image,
        brightness=sample.brightness,
        label=np.int64(label),
        label_name=label_name,
        source_path=str(sample.source_path),
        frame_stem=sample.frame_stem,
        candidate_index=np.int64(sample.candidate_index),
        bbox_xywh=np.asarray(sample.bbox_xywh, dtype=np.int32),
        square_crop_xywh=np.asarray(sample.square_crop_xywh, dtype=np.int32),
        expansion_factor=np.float32(expansion_factor),
        output_size=np.asarray(output_size, dtype=np.int32),
    )

    if preview_enabled:
        preview = build_preview_image(sample.image, sample.brightness)
        preview_path = (output_dir / preview_dir / sample_stem).with_suffix(preview_ext)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), preview)


def main() -> None:
    args = parse_args()

    annotations_path = Path(args.annotations).expanduser()
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    annotations_by_frame, predictions_path = load_annotations(annotations_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions meta not found: {predictions_path}")

    meta = json.loads(predictions_path.read_text(encoding="utf-8"))
    if meta.get("frame_data_format") != "ndjson":
        raise ValueError("Predictions meta must reference ndjson frame data.")

    frames_dir = Path(meta["frames_dir"]).expanduser()
    diff_frames_dir = Path(meta["diff_frames_dir"]).expanduser()
    candidates_dir = Path(meta["candidates_dir"]).expanduser()
    frames_ndjson = predictions_path.parent / "frames.ndjson"
    if not frames_ndjson.is_file():
        raise FileNotFoundError(f"frames.ndjson not found: {frames_ndjson}")

    if args.output_root is None:
        processing_root = frames_dir.parent.parent
        output_root = processing_root / "ball_datasets" / frames_dir.name
    else:
        output_root = Path(args.output_root).expanduser()

    train_ball_dir = output_root / "train" / "_ball"
    train_not_ball_dir = output_root / "train" / "_ball not"
    val_ball_dir = output_root / "val" / "_ball"
    val_not_ball_dir = output_root / "val" / "_ball not"

    for root in (train_ball_dir, train_not_ball_dir, val_ball_dir, val_not_ball_dir):
        ensure_output_dirs(
            root=root,
            preview_enabled=args.preview,
            preview_dir="jpg",
            npz_dir="npz",
        )

    config = AnnotatedCandidateDatasetConfig(
        annotated_ball_dir=frames_dir,
        annotated_not_ball_dir=frames_dir,
        output_ball_dir=train_ball_dir,
        output_not_ball_dir=train_not_ball_dir,
        frames_dir=frames_dir,
        candidates_dir=candidates_dir,
        diff_frames_dir=diff_frames_dir,
        expansion_factor=float(args.expansion_factor),
        output_size=(int(args.output_size[0]), int(args.output_size[1])),
        padding_mode=str(args.padding_mode),
        preview_enabled=bool(args.preview),
        preview_ext=str(args.preview_ext),
        strict_filename_parsing=True,
        flatten_output=True,
        npz_dir_name="npz",
        preview_dir_name="jpg",
    )
    preprocessor = CandidateCropPreprocessor(config)

    train_frames, val_frames = build_split_sets(
        frame_names=sorted(annotations_by_frame.keys()),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    summary = DatasetBuildSummary()
    split_counts = {
        "train": {"ball": 0, "not_ball": 0},
        "val": {"ball": 0, "not_ball": 0},
    }

    with frames_ndjson.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame_entry = json.loads(line)
            frame_name = frame_entry.get("frame_name")
            if not isinstance(frame_name, str) or not frame_name:
                frame_value = frame_entry.get("frame")
                frame_name = Path(frame_value).name if frame_value else ""
            if frame_name not in annotations_by_frame:
                continue

            annotation = annotations_by_frame[frame_name]
            if annotation.get("label_source") == "none":
                continue

            split = "val" if frame_name in val_frames else "train"
            if split == "val":
                ball_dir = val_ball_dir
                not_ball_dir = val_not_ball_dir
            else:
                ball_dir = train_ball_dir
                not_ball_dir = train_not_ball_dir

            candidates = frame_entry.get("candidates") or []
            candidate_map = {
                candidate.get("candidate_uid"): candidate
                for candidate in candidates
                if isinstance(candidate, dict)
            }
            selected_uid = annotation.get("selected_ball_candidate_uid")
            no_ball = bool(annotation.get("no_ball"))

            if no_ball:
                labeled_candidates = [(candidate, 0) for candidate in candidates]
            elif selected_uid:
                selected_candidate = candidate_map.get(selected_uid)
                if selected_candidate is None:
                    summary.skipped += 1
                    summary.skip_reasons["missing_selected_candidate"] += 1
                    continue
                labeled_candidates = []
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    label = 1 if candidate.get("candidate_uid") == selected_uid else 0
                    labeled_candidates.append((candidate, label))
            else:
                summary.skipped += 1
                summary.skip_reasons["missing_label_selection"] += 1
                continue

            for candidate, label in labeled_candidates:
                candidate_index = candidate.get("candidate_index")
                try:
                    candidate_index = int(candidate_index)
                except (TypeError, ValueError):
                    summary.skipped += 1
                    summary.skip_reasons["invalid_candidate_index"] += 1
                    continue
                frame_stem = Path(frame_name).stem
                dummy_path = frames_dir / f"{frame_stem}_candidate_{candidate_index:03d}.jpg"
                sample, reason = preprocessor.build_processed_sample(dummy_path)
                if sample is None:
                    summary.skipped += 1
                    summary.skip_reasons[reason or "failed_preprocess"] += 1
                    continue

                sample_stem = sanitize_token(candidate.get("candidate_uid") or dummy_path.stem)
                if not sample_stem:
                    sample_stem = sanitize_token(dummy_path.stem) or "sample"

                if label == 1:
                    save_sample(
                        output_dir=ball_dir,
                        sample_stem=sample_stem,
                        sample=sample,
                        label=1,
                        label_name="ball",
                        expansion_factor=config.expansion_factor,
                        output_size=config.output_size,
                        preview_enabled=args.preview,
                        preview_dir="jpg",
                        preview_ext=str(args.preview_ext),
                        npz_dir="npz",
                    )
                    split_counts[split]["ball"] += 1
                    summary.processed_ball += 1
                else:
                    save_sample(
                        output_dir=not_ball_dir,
                        sample_stem=sample_stem,
                        sample=sample,
                        label=0,
                        label_name="ball_not",
                        expansion_factor=config.expansion_factor,
                        output_size=config.output_size,
                        preview_enabled=args.preview,
                        preview_dir="jpg",
                        preview_ext=str(args.preview_ext),
                        npz_dir="npz",
                    )
                    split_counts[split]["not_ball"] += 1
                    summary.processed_not_ball += 1

    print("=== Annotation Dataset Summary ===")
    print(
        f"Train: ball={split_counts['train']['ball']} "
        f"not_ball={split_counts['train']['not_ball']}"
    )
    print(
        f"Val: ball={split_counts['val']['ball']} "
        f"not_ball={split_counts['val']['not_ball']}"
    )
    print(f"Skipped: {summary.skipped}")
    if summary.skip_reasons:
        print("Skip reasons:")
        for reason, count in summary.skip_reasons.most_common():
            print(f"  - {reason}: {count}")

    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
