from __future__ import annotations

from pathlib import Path

import cv2


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

