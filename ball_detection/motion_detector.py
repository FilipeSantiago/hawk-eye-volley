from __future__ import annotations

import json
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class MotionDetector:
    def __init__(
        self,
        frames_dir: Path,
        annotated_dir: Path,
        candidates_dir: Path,
        min_moving_area: int,
        max_moving_area: int | None = None,
        diff_threshold: int = 12,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.annotated_dir = Path(annotated_dir)
        self.candidates_dir = Path(candidates_dir)
        self.min_moving_area = int(min_moving_area)
        self.max_moving_area = (
            None if max_moving_area is None else int(max_moving_area)
        )
        self.diff_threshold = int(diff_threshold)

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
            prev = cv2.imread(str(prev_path))
            curr = cv2.imread(str(curr_path))
            if prev is None or curr is None:
                continue

            gray, mask = self._motion_mask(prev, curr, self.diff_threshold)
            boxes = self._extract_boxes(mask, gray)
            vis = self._draw_boxes(curr, boxes)

            cv2.imwrite(str(self.annotated_dir / curr_path.name), vis)
            self._write_candidates_json(
                self.candidates_dir / f"{curr_path.stem}.json",
                curr_path.name,
                idx + 1,
                boxes,
            )
            processed += 1

        return processed

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

    def _extract_boxes(self, mask, gray) -> list[tuple[int, int, int, int, int, float]]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes: list[tuple[int, int, int, int, int, float]] = []
        gray_h, gray_w = gray.shape[:2]
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
            gray_roi = gray[y1:y2, x1:x2]
            gray_mean = float(gray_roi.mean()) if gray_roi.size else 0.0
            boxes.append((int(x), int(y), int(w), int(h), int(area), gray_mean))
        return boxes

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
    def _motion_mask(prev_frame, curr_frame, threshold: int):
        diff = cv2.absdiff(curr_frame, prev_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return gray, mask

    @staticmethod
    def _draw_boxes(image, boxes, color=(0, 255, 255), thickness: int = 2):
        vis = image.copy()
        height, width = vis.shape[:2]
        for x, y, w, h, *_ in boxes:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x + w)
            y2 = min(height - 1, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        return vis

    @staticmethod
    def _write_candidates_json(path: Path, frame_name: str, index: int, boxes) -> None:
        payload = {
            "frame": frame_name,
            "index": index,
            "num_candidates": len(boxes),
            "candidates": [
                {"bbox": [x, y, w, h], "area": area, "gray_mean": gray_mean}
                for x, y, w, h, area, gray_mean in boxes
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
