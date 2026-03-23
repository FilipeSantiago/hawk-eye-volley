from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ball_detection.annotated_candidate_dataset_config import AnnotatedCandidateDatasetConfig
from ball_detection.candidate_dataset_constants import (
    FILENAME_PATTERN,
    IMAGE_EXTENSIONS,
    PADDING_MODE_TO_CV2,
)
from ball_detection.processed_candidate_sample import ProcessedCandidateSample


class CandidateCropPreprocessor:
    """Prepare normalized RGB and brightness crops from original frame artifacts."""

    def __init__(self, config: AnnotatedCandidateDatasetConfig) -> None:
        self.config = config
        self._frame_map = self._index_images(self.config.frames_dir, "frame")
        self._diff_map = self._index_images(self.config.diff_frames_dir, "diff")
        self._candidate_json_map = self._index_candidate_json_files(self.config.candidates_dir)
        self._candidate_payload_cache: dict[str, dict[str, Any]] = {}

    def parse_candidate_filename(self, stem: str) -> tuple[str, int] | None:
        """Extract (frame_stem, candidate_index_1_based) from crop filename stem."""
        if self.config.strict_filename_parsing:
            match = FILENAME_PATTERN.fullmatch(stem)
        else:
            match = FILENAME_PATTERN.search(stem)
        if not match:
            return None

        frame_stem = match.group(1)
        try:
            candidate_index_1_based = int(match.group(2))
        except ValueError:
            return None

        return frame_stem, candidate_index_1_based

    def load_candidate_bbox(
        self, frame_stem: str, candidate_index_0_based: int
    ) -> tuple[int, int, int, int] | None:
        """Load candidate bbox `[x, y, w, h]` from frame JSON."""
        payload = self._load_candidate_payload(frame_stem)
        if payload is None:
            return None

        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            return None
        if candidate_index_0_based < 0 or candidate_index_0_based >= len(candidates):
            return None

        candidate = candidates[candidate_index_0_based]
        if not isinstance(candidate, dict):
            return None
        bbox = candidate.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None

        try:
            x, y, w, h = [int(round(float(v))) for v in bbox]
        except (TypeError, ValueError):
            return None

        if w <= 0 or h <= 0:
            return None

        return x, y, w, h

    def find_original_frame(self, frame_stem: str) -> Path | None:
        """Resolve original RGB frame path for frame stem."""
        return self._frame_map.get(frame_stem)

    def find_diff_frame(self, frame_stem: str) -> Path | None:
        """Resolve diff frame path for frame stem."""
        preferred = f"diff_{frame_stem}"
        if preferred in self._diff_map:
            return self._diff_map[preferred]
        return self._diff_map.get(frame_stem)

    def compute_square_crop_region(
        self,
        bbox_xywh: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        """Compute expanded square crop around bbox center."""
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            return None

        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        side = max(w, h) * self.config.expansion_factor
        side_i = int(round(side))
        if side_i <= 0:
            return None

        x1 = int(round(cx - (side_i / 2.0)))
        y1 = int(round(cy - (side_i / 2.0)))
        return x1, y1, side_i, side_i

    def crop_with_padding(self, image: np.ndarray, crop_xywh: tuple[int, int, int, int]) -> np.ndarray | None:
        """Crop an image region with safe border padding when out of bounds."""
        x, y, w, h = crop_xywh
        if w <= 0 or h <= 0:
            return None

        img_h, img_w = image.shape[:2]
        pad_left = max(0, -x)
        pad_top = max(0, -y)
        pad_right = max(0, (x + w) - img_w)
        pad_bottom = max(0, (y + h) - img_h)

        border_type = PADDING_MODE_TO_CV2[self.config.padding_mode]
        if pad_left or pad_top or pad_right or pad_bottom:
            if border_type == cv2.BORDER_CONSTANT:
                padded = cv2.copyMakeBorder(
                    image,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    border_type,
                    value=0,
                )
            else:
                padded = cv2.copyMakeBorder(
                    image,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    border_type,
                )
            start_x = x + pad_left
            start_y = y + pad_top
        else:
            padded = image
            start_x = x
            start_y = y

        crop = padded[start_y : start_y + h, start_x : start_x + w]
        if crop.size == 0:
            return None
        if crop.shape[0] != h or crop.shape[1] != w:
            return None
        return crop

    def resize_pair(
        self, rgb_crop: np.ndarray, diff_crop: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize RGB and diff crops to fixed output size."""
        out_w, out_h = self.config.output_size
        rgb_resized = cv2.resize(rgb_crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        diff_resized = cv2.resize(diff_crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return rgb_resized, diff_resized

    def build_processed_sample(
        self,
        source_path: Path,
    ) -> tuple[ProcessedCandidateSample | None, str | None]:
        """Build a processed sample from an annotated crop path."""
        parsed = self.parse_candidate_filename(source_path.stem)
        if parsed is None:
            return None, "filename_parse_failed"

        frame_stem, candidate_index_1_based = parsed
        candidate_index_0_based = candidate_index_1_based - 1
        if candidate_index_0_based < 0:
            return None, "invalid_candidate_index"

        bbox_xywh = self.load_candidate_bbox(frame_stem, candidate_index_0_based)
        if bbox_xywh is None:
            return None, "missing_or_invalid_bbox"

        square_crop_xywh = self.compute_square_crop_region(bbox_xywh)
        if square_crop_xywh is None:
            return None, "invalid_square_crop_region"

        frame_path = self.find_original_frame(frame_stem)
        if frame_path is None:
            return None, "missing_original_frame"

        diff_path = self.find_diff_frame(frame_stem)
        if diff_path is None:
            return None, "missing_diff_frame"

        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None, "failed_to_read_original_frame"

        diff_raw = cv2.imread(str(diff_path), cv2.IMREAD_UNCHANGED)
        if diff_raw is None:
            return None, "failed_to_read_diff_image"

        if diff_raw.ndim == 2:
            diff_gray = diff_raw
        elif diff_raw.ndim == 3:
            diff_gray = cv2.cvtColor(diff_raw, cv2.COLOR_BGR2GRAY)
        else:
            return None, "invalid_diff_image_shape"

        rgb_crop_bgr = self.crop_with_padding(frame_bgr, square_crop_xywh)
        if rgb_crop_bgr is None:
            return None, "failed_rgb_crop_with_padding"

        diff_crop = self.crop_with_padding(diff_gray, square_crop_xywh)
        if diff_crop is None:
            return None, "failed_diff_crop_with_padding"

        rgb_crop_bgr, diff_crop = self.resize_pair(rgb_crop_bgr, diff_crop)
        rgb_crop = cv2.cvtColor(rgb_crop_bgr, cv2.COLOR_BGR2RGB)

        sample = ProcessedCandidateSample(
            image=rgb_crop,
            brightness=diff_crop,
            source_path=source_path,
            frame_stem=frame_stem,
            candidate_index=candidate_index_0_based,
            bbox_xywh=bbox_xywh,
            square_crop_xywh=square_crop_xywh,
        )
        return sample, None

    def _index_images(self, root_dir: Path, role_name: str) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        if not root_dir.is_dir():
            print(f"[WARN] {role_name}_dir not found: {root_dir}")
            return mapping

        for path in sorted(root_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = path.stem
            if stem in mapping and mapping[stem] != path:
                print(
                    f"[WARN] Duplicate {role_name} image for {stem}: "
                    f"using {mapping[stem]}, ignoring {path}"
                )
                continue
            mapping[stem] = path
        return mapping

    def _index_candidate_json_files(self, candidates_dir: Path) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        if not candidates_dir.is_dir():
            print(f"[WARN] candidates_dir not found: {candidates_dir}")
            return mapping

        for path in sorted(candidates_dir.rglob("*.json")):
            stem = path.stem
            if stem in mapping and mapping[stem] != path:
                print(
                    f"[WARN] Duplicate candidate JSON for {stem}: "
                    f"using {mapping[stem]}, ignoring {path}"
                )
                continue
            mapping[stem] = path
        return mapping

    def _load_candidate_payload(self, frame_stem: str) -> dict[str, Any] | None:
        if frame_stem in self._candidate_payload_cache:
            return self._candidate_payload_cache[frame_stem]

        json_path = self._candidate_json_map.get(frame_stem)
        if json_path is None:
            return None

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        self._candidate_payload_cache[frame_stem] = payload
        return payload

