from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ball_detection.annotated_candidate_dataset_config import AnnotatedCandidateDatasetConfig
from ball_detection.candidate_crop_preprocessor import CandidateCropPreprocessor
from ball_detection.candidate_dataset_constants import IMAGE_EXTENSIONS, TOKEN_SANITIZER_PATTERN
from ball_detection.dataset_build_summary import DatasetBuildSummary


class AnnotatedCandidateDatasetBuilder:
    """Build NPZ training dataset from annotated candidate images."""

    def __init__(self, config: AnnotatedCandidateDatasetConfig) -> None:
        self.config = config
        self.summary = DatasetBuildSummary()
        self.preprocessor = CandidateCropPreprocessor(config)

    def run(self) -> DatasetBuildSummary:
        self.config.output_ball_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_not_ball_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_output_roots(self.config.output_ball_dir)
        self._ensure_output_roots(self.config.output_not_ball_dir)

        self._process_labeled_dir(
            source_dir=self.config.annotated_ball_dir,
            output_dir=self.config.output_ball_dir,
            label=1,
            label_name="ball",
        )
        self._process_labeled_dir(
            source_dir=self.config.annotated_not_ball_dir,
            output_dir=self.config.output_not_ball_dir,
            label=0,
            label_name="ball_not",
        )

        self._print_summary()
        return self.summary

    def _process_labeled_dir(
        self,
        source_dir: Path,
        output_dir: Path,
        label: int,
        label_name: str,
    ) -> None:
        if not source_dir.is_dir():
            self._warn_skip(source_dir, "missing_annotated_dir")
            return

        for source_path in sorted(source_dir.rglob("*")):
            if not source_path.is_file():
                continue
            if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            sample, reason = self.preprocessor.build_processed_sample(source_path)
            if sample is None:
                self._warn_skip(source_path, reason or "unknown_preprocess_failure")
                continue

            relative_base = self._make_relative_base(
                source_path=source_path,
                source_root=source_dir,
            )
            npz_path = (
                output_dir / self.config.npz_dir_name / relative_base
            ).with_suffix(".npz")
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
                expansion_factor=np.float32(self.config.expansion_factor),
                output_size=np.asarray(self.config.output_size, dtype=np.int32),
            )

            if self.config.preview_enabled:
                preview = self._build_preview_image(sample.image, sample.brightness)
                preview_path = (
                    output_dir / self.config.preview_dir_name / relative_base
                ).with_suffix(self.config.preview_ext)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview_path), preview)

            if label == 1:
                self.summary.processed_ball += 1
            else:
                self.summary.processed_not_ball += 1

    def _make_relative_base(self, source_path: Path, source_root: Path) -> Path:
        rel_no_ext = source_path.relative_to(source_root).with_suffix("")

        if self.config.flatten_output:
            tokens = [self._sanitize_token(part) for part in rel_no_ext.parts]
            safe_stem = "__".join(token for token in tokens if token) or "sample"
            return Path(safe_stem)

        safe_parts = [self._sanitize_token(part) for part in rel_no_ext.parts]
        if not safe_parts:
            safe_parts = ["sample"]
        return Path(*safe_parts)

    def _ensure_output_roots(self, output_dir: Path) -> None:
        (output_dir / self.config.npz_dir_name).mkdir(parents=True, exist_ok=True)
        if self.config.preview_enabled:
            (output_dir / self.config.preview_dir_name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_token(value: str) -> str:
        sanitized = TOKEN_SANITIZER_PATTERN.sub("_", value.strip())
        return sanitized.strip("_")

    @staticmethod
    def _build_preview_image(rgb_image: np.ndarray, brightness: np.ndarray) -> np.ndarray:
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        brightness_vis = cv2.cvtColor(brightness, cv2.COLOR_GRAY2BGR)
        if brightness_vis.shape[:2] != rgb_bgr.shape[:2]:
            brightness_vis = cv2.resize(
                brightness_vis,
                (rgb_bgr.shape[1], rgb_bgr.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.hconcat([rgb_bgr, brightness_vis])

    def _warn_skip(self, source_path: Path, reason: str) -> None:
        self.summary.skipped += 1
        self.summary.skip_reasons[reason] += 1
        print(f"[WARN] Skip {source_path}: {reason}")

    def _print_summary(self) -> None:
        print("=== AnnotatedCandidateDatasetBuilder Summary ===")
        print(f"Processed ball samples: {self.summary.processed_ball}")
        print(f"Processed not-ball samples: {self.summary.processed_not_ball}")
        print(f"Skipped samples: {self.summary.skipped}")
        if self.summary.skip_reasons:
            print("Skip reasons:")
            for reason, count in self.summary.skip_reasons.most_common():
                print(f"  - {reason}: {count}")
