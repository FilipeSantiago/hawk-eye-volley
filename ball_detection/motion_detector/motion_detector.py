from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class MotionCenterConfig:
    """Configuration for estimating the current ball center from motion blobs."""

    # Strategy: "midpoint" or "appearance_weighted".
    strategy: str = "midpoint"

    # Connected component support source: "union" or "sum".
    component_source: str = "union"

    # Optional override for the diff threshold used to build M_prev/M_next.
    diff_threshold: int | None = None

    # If component area is tiny/noisy, fall back to raw centroid.
    min_center_area: int = 12

    # Candidate bbox sizing.
    bbox_size: int | None = None  # If set, use this fixed square size.
    bbox_scale: float = 0.9  # Used with sqrt(area) when bbox_size is None.
    bbox_min_size: int = 8
    bbox_max_size: int = 80

    # Appearance weighting parameters.
    appearance_padding: int = 4
    appearance_weight_mode: str = "contrast"  # "contrast" or "brightness"
    appearance_blur_ksize: tuple[int, int] = (3, 3)
    appearance_gamma: float = 1.0
    motion_weight_scale: float = 0.5
    appearance_blend: float = 0.7


class MotionCenterEstimator:
    """Estimate the current ball center inside a motion component."""

    def __init__(self, config: MotionCenterConfig) -> None:
        self.config = config

    def estimate_component(
        self,
        frame_gray: np.ndarray,
        diff_prev: np.ndarray,
        diff_next: np.ndarray,
        component_bbox: tuple[int, int, int, int],
        component_mask: np.ndarray,
        prev_support: np.ndarray,
        next_support: np.ndarray,
        component_area: int,
        raw_centroid: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[int, int, int, int], dict[str, Any]]:
        """Estimate center and candidate bbox for a single connected component."""

        x, y, w, h = component_bbox
        diff_prev_roi = diff_prev[y : y + h, x : x + w]
        diff_next_roi = diff_next[y : y + h, x : x + w]

        centroid_prev = self._weighted_centroid(prev_support, diff_prev_roi, (x, y))
        centroid_next = self._weighted_centroid(next_support, diff_next_roi, (x, y))

        mean_prev = self._masked_mean(diff_prev_roi, component_mask)
        mean_next = self._masked_mean(diff_next_roi, component_mask)

        midpoint, midpoint_source = self._resolve_midpoint(
            centroid_prev, centroid_next, raw_centroid
        )

        strategy_used = midpoint_source
        estimated_center = midpoint

        if component_area < max(1, int(self.config.min_center_area)):
            strategy_used = "raw_centroid_tiny_component"
            estimated_center = raw_centroid
        elif self.config.strategy == "appearance_weighted":
            appearance_center = self._appearance_weighted_center(
                frame_gray=frame_gray,
                diff_prev=diff_prev,
                diff_next=diff_next,
                component_bbox=component_bbox,
                component_mask=component_mask,
            )
            if appearance_center is None:
                strategy_used = f"{midpoint_source}_fallback"
                estimated_center = midpoint
            else:
                blend = float(self.config.appearance_blend)
                if blend < 0.0:
                    blend = 0.0
                if blend > 1.0:
                    blend = 1.0
                estimated_center = (
                    (1.0 - blend) * midpoint[0] + blend * appearance_center[0],
                    (1.0 - blend) * midpoint[1] + blend * appearance_center[1],
                )
                strategy_used = "appearance_weighted"
        elif self.config.strategy != "midpoint":
            strategy_used = f"unknown_strategy_{self.config.strategy}"
            estimated_center = midpoint

        candidate_bbox = self._candidate_bbox(
            estimated_center,
            component_area,
            frame_gray.shape[:2],
        )

        debug = {
            "component_bbox": [int(x), int(y), int(w), int(h)],
            "component_centroid": [float(raw_centroid[0]), float(raw_centroid[1])],
            "centroid_prev": self._serialize_point(centroid_prev),
            "centroid_next": self._serialize_point(centroid_next),
            "estimated_center": [float(estimated_center[0]), float(estimated_center[1])],
            "strategy_used": strategy_used,
            "component_area": int(component_area),
            "mean_diff_prev": float(mean_prev),
            "mean_diff_next": float(mean_next),
        }
        return estimated_center, candidate_bbox, debug

    def _appearance_weighted_center(
        self,
        frame_gray: np.ndarray,
        diff_prev: np.ndarray,
        diff_next: np.ndarray,
        component_bbox: tuple[int, int, int, int],
        component_mask: np.ndarray,
    ) -> tuple[float, float] | None:
        x, y, w, h = component_bbox
        pad = max(0, int(self.config.appearance_padding))
        height, width = frame_gray.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        if x2 <= x1 or y2 <= y1:
            return None

        gray_roi = frame_gray[y1:y2, x1:x2]
        if gray_roi.size == 0:
            return None

        kx, ky = self.config.appearance_blur_ksize
        if kx > 0 and ky > 0:
            gray_roi = cv2.GaussianBlur(gray_roi, (int(kx), int(ky)), 0)

        weights = gray_roi.astype(np.float32) / 255.0
        mode = str(self.config.appearance_weight_mode).lower().strip()
        if mode == "contrast":
            mean_val = float(weights.mean()) if weights.size else 0.0
            weights = np.abs(weights - mean_val)
        elif mode != "brightness":
            weights = np.abs(weights - float(weights.mean()) if weights.size else 0.0)

        gamma = float(self.config.appearance_gamma)
        if gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
            weights = np.power(weights, gamma)

        motion_scale = float(self.config.motion_weight_scale)
        if motion_scale > 0.0:
            motion_roi = (
                diff_prev[y1:y2, x1:x2].astype(np.float32)
                + diff_next[y1:y2, x1:x2].astype(np.float32)
            )
            if motion_roi.size:
                motion_norm = motion_roi / (float(motion_roi.max()) + 1e-6)
                weights = weights * (1.0 + motion_scale * motion_norm)

        mask_padded = np.zeros_like(weights, dtype=bool)
        offset_x = x - x1
        offset_y = y - y1
        mask_padded[
            offset_y : offset_y + component_mask.shape[0],
            offset_x : offset_x + component_mask.shape[1],
        ] = component_mask

        return self._weighted_centroid(mask_padded, weights, (x1, y1))

    @staticmethod
    def _serialize_point(point: tuple[float, float] | None) -> list[float] | None:
        if point is None:
            return None
        return [float(point[0]), float(point[1])]

    @staticmethod
    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if values.size == 0 or mask.size == 0:
            return 0.0
        masked = values[mask]
        if masked.size == 0:
            return 0.0
        return float(masked.mean())

    @staticmethod
    def _weighted_centroid(
        mask: np.ndarray,
        weights: np.ndarray,
        offset: tuple[int, int],
    ) -> tuple[float, float] | None:
        if mask.size == 0 or weights.size == 0:
            return None
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        w = weights[ys, xs].astype(np.float32)
        w_sum = float(w.sum())
        if w_sum <= 1e-6:
            return None
        cx = float((xs * w).sum() / w_sum) + float(offset[0])
        cy = float((ys * w).sum() / w_sum) + float(offset[1])
        return cx, cy

    @staticmethod
    def _resolve_midpoint(
        centroid_prev: tuple[float, float] | None,
        centroid_next: tuple[float, float] | None,
        raw_centroid: tuple[float, float],
    ) -> tuple[tuple[float, float], str]:
        if centroid_prev is not None and centroid_next is not None:
            midpoint = (
                (centroid_prev[0] + centroid_next[0]) * 0.5,
                (centroid_prev[1] + centroid_next[1]) * 0.5,
            )
            return midpoint, "midpoint"
        if centroid_prev is not None:
            return centroid_prev, "centroid_prev_only"
        if centroid_next is not None:
            return centroid_next, "centroid_next_only"
        return raw_centroid, "raw_centroid"

    def _candidate_bbox(
        self,
        center: tuple[float, float],
        component_area: int,
        frame_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape
        size = self._resolve_bbox_size(component_area, width, height)
        half = size / 2.0
        x = int(round(center[0] - half))
        y = int(round(center[1] - half))
        x = max(0, min(width - size, x)) if width > size else 0
        y = max(0, min(height - size, y)) if height > size else 0
        return int(x), int(y), int(size), int(size)

    def _resolve_bbox_size(self, component_area: int, width: int, height: int) -> int:
        if self.config.bbox_size is not None:
            size = int(self.config.bbox_size)
        else:
            area = max(1, int(component_area))
            size = int(round(math.sqrt(area) * float(self.config.bbox_scale)))

        size = max(int(self.config.bbox_min_size), size)
        size = min(int(self.config.bbox_max_size), size)
        size = min(size, width, height)
        return max(1, size)


class MotionDetector:
    def __init__(
        self,
        frames_dir: Path,
        annotated_dir: Path,
        candidates_dir: Path,
        min_moving_area: int,
        max_moving_area: int | None = None,
        diff_threshold: int = 12,
        center_config: MotionCenterConfig | None = None,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.annotated_dir = Path(annotated_dir)
        self.candidates_dir = Path(candidates_dir)
        self.min_moving_area = int(min_moving_area)
        self.max_moving_area = (
            None if max_moving_area is None else int(max_moving_area)
        )
        self.diff_threshold = int(diff_threshold)
        self.center_config = center_config or MotionCenterConfig()
        self.center_estimator = MotionCenterEstimator(self.center_config)

    def run_motion_detector(self) -> int:
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = self._iter_frames()
        if len(frame_paths) < 2:
            raise FileNotFoundError(f"Need at least 2 frames in: {self.frames_dir}")

        processed = 0
        for idx in range(1, len(frame_paths)):
            prev_path = frame_paths[idx - 1]
            curr_path = frame_paths[idx]
            next_path = frame_paths[idx + 1] if idx + 1 < len(frame_paths) else None
            prev = cv2.imread(str(prev_path))
            curr = cv2.imread(str(curr_path))
            next_frame = cv2.imread(str(next_path)) if next_path is not None else None
            if prev is None or curr is None:
                continue

            candidates = self.estimate_candidates_for_triplet(prev, curr, next_frame)
            vis = self._draw_boxes(curr, candidates)

            cv2.imwrite(str(self.annotated_dir / curr_path.name), vis)
            self._write_candidates_json(
                self.candidates_dir / f"{curr_path.stem}.json",
                curr_path.name,
                idx + 1,
                candidates,
            )
            processed += 1

        return processed

    def estimate_candidates_for_triplet(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        next_frame: np.ndarray | None,
    ) -> list[dict[str, Any]]:
        """Estimate motion candidates from a (prev, curr, next) frame triplet."""
        threshold = (
            int(self.center_config.diff_threshold)
            if self.center_config.diff_threshold is not None
            else self.diff_threshold
        )

        diff_prev = self._diff_gray(prev_frame, curr_frame)
        diff_next = (
            self._diff_gray(curr_frame, next_frame)
            if next_frame is not None
            else np.zeros_like(diff_prev)
        )

        mask_prev = self._threshold_mask(diff_prev, threshold)
        mask_next = (
            self._threshold_mask(diff_next, threshold)
            if next_frame is not None
            else np.zeros_like(mask_prev)
        )

        diff_sum = diff_prev.astype(np.float32) + diff_next.astype(np.float32)
        source = str(self.center_config.component_source).lower().strip()
        if source == "sum":
            component_mask = self._threshold_mask(diff_sum, threshold)
        else:
            component_mask = cv2.bitwise_or(mask_prev, mask_next)

        return self._extract_candidates(
            component_mask=component_mask,
            diff_prev=diff_prev,
            diff_next=diff_next,
            mask_prev=mask_prev,
            mask_next=mask_next,
            diff_sum=diff_sum,
            frame_cur=curr_frame,
        )

    def export_cropped_candidates(
        self,
        output_dir: Path | None = None,
        image_ext: str = ".jpg",
        padding: int = 0,
    ) -> int:
        if not image_ext.startswith("."):
            image_ext = f".{image_ext}"

        crops_dir = Path(output_dir) if output_dir is not None else self.candidates_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        candidate_files = sorted(self.candidates_dir.glob("*.json"))
        for candidate_path in candidate_files:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            frame_name = payload.get("frame")
            frame_path = self._resolve_frame_path(frame_name, candidate_path.stem)
            if frame_path is None:
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            frame_h, frame_w = frame.shape[:2]
            frame_crop_dir = crops_dir / frame_path.stem
            if frame_crop_dir.exists():
                for old_crop in frame_crop_dir.glob("*_candidate_*"):
                    if old_crop.is_file():
                        old_crop.unlink()
            frame_crop_dir.mkdir(parents=True, exist_ok=True)

            candidates = payload.get("candidates", [])
            if not isinstance(candidates, list):
                continue

            for i, candidate in enumerate(candidates, start=1):
                bbox = candidate.get("bbox") if isinstance(candidate, dict) else None
                if not bbox or len(bbox) != 4:
                    continue

                x, y, w, h = [int(round(v)) for v in bbox]
                if w <= 0 or h <= 0:
                    continue

                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame_w, x + w + padding)
                y2 = min(frame_h, y + h + padding)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                crop_path = frame_crop_dir / f"{frame_path.stem}_candidate_{i:03d}{image_ext}"
                saved = cv2.imwrite(str(crop_path), crop)
                if not saved:
                    raise RuntimeError(f"Failed to write candidate crop: {crop_path}")
                exported += 1

        return exported

    def _iter_frames(self) -> list[Path]:
        return sorted(
            path
            for path in self.frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _extract_candidates(
        self,
        component_mask: np.ndarray,
        diff_prev: np.ndarray,
        diff_next: np.ndarray,
        mask_prev: np.ndarray,
        mask_next: np.ndarray,
        diff_sum: np.ndarray,
        frame_cur: np.ndarray,
    ) -> list[dict[str, Any]]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            component_mask, connectivity=8
        )
        candidates: list[dict[str, Any]] = []
        gray_h, gray_w = diff_prev.shape[:2]
        frame_gray = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < self.min_moving_area:
                continue
            if self.max_moving_area is not None and area > self.max_moving_area:
                continue
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(gray_w, int(x + w))
            y2 = min(gray_h, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            component_mask_roi = labels[y1:y2, x1:x2] == i
            if not np.any(component_mask_roi):
                continue

            raw_centroid = centroids[i]
            raw_cx = float(raw_centroid[0])
            raw_cy = float(raw_centroid[1])
            if not np.isfinite(raw_cx) or not np.isfinite(raw_cy):
                ys, xs = np.nonzero(component_mask_roi)
                if ys.size == 0:
                    continue
                raw_cx = float(xs.mean() + x1)
                raw_cy = float(ys.mean() + y1)

            prev_support = component_mask_roi & (mask_prev[y1:y2, x1:x2] > 0)
            next_support = component_mask_roi & (mask_next[y1:y2, x1:x2] > 0)

            # The diff blob often spans previous + current ball positions.
            # Estimate the current center near the temporal middle rather than the raw centroid.
            estimated_center, candidate_bbox, debug = self.center_estimator.estimate_component(
                frame_gray=frame_gray,
                diff_prev=diff_prev,
                diff_next=diff_next,
                component_bbox=(x1, y1, int(x2 - x1), int(y2 - y1)),
                component_mask=component_mask_roi,
                prev_support=prev_support,
                next_support=next_support,
                component_area=int(area),
                raw_centroid=(raw_cx, raw_cy),
            )

            diff_sum_roi = diff_sum[y1:y2, x1:x2]
            gray_mean = (
                float(diff_sum_roi[component_mask_roi].mean())
                if diff_sum_roi.size
                else 0.0
            )

            bx, by, bw, bh = candidate_bbox
            candidate: dict[str, Any] = {
                "bbox": [int(bx), int(by), int(bw), int(bh)],
                "center": [float(estimated_center[0]), float(estimated_center[1])],
                "area": int(area),
                "gray_mean": float(gray_mean),
            }
            candidate.update(debug)
            candidates.append(candidate)

        return candidates

    def _resolve_frame_path(self, frame_name: object, frame_stem: str) -> Path | None:
        if isinstance(frame_name, str) and frame_name:
            candidate = self.frames_dir / frame_name
            if candidate.is_file():
                return candidate

        for ext in IMAGE_EXTENSIONS:
            candidate = self.frames_dir / f"{frame_stem}{ext}"
            if candidate.is_file():
                return candidate

        return None

    @staticmethod
    def _diff_gray(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(curr_frame, prev_frame)
        return cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _threshold_mask(diff: np.ndarray, threshold: int) -> np.ndarray:
        if diff.dtype != np.uint8:
            return (diff >= threshold).astype(np.uint8) * 255
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def _draw_boxes(image, boxes, color=(0, 255, 255), thickness: int = 2):
        vis = image.copy()
        height, width = vis.shape[:2]
        for item in boxes:
            if isinstance(item, dict):
                bbox = item.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = [int(round(v)) for v in bbox]
            else:
                x, y, w, h, *_ = item
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(width - 1, int(x + w))
            y2 = min(height - 1, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        return vis

    @staticmethod
    def _write_candidates_json(
        path: Path,
        frame_name: str,
        index: int,
        candidates: Iterable[dict[str, Any]],
    ) -> None:
        candidate_list = list(candidates)
        payload = {
            "frame": frame_name,
            "index": index,
            "num_candidates": len(candidate_list),
            "candidates": candidate_list,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
