from __future__ import annotations

from pathlib import Path

from ball_detection.training.annotated_candidate_dataset_builder import (
    AnnotatedCandidateDatasetBuilder,
)
from ball_detection.training.annotated_candidate_dataset_config import (
    AnnotatedCandidateDatasetConfig,
)


def main() -> None:
    config = AnnotatedCandidateDatasetConfig(
        annotated_ball_dir=Path("/home/skynet/Downloads/hawk eye/data/ball/"),
        annotated_not_ball_dir=Path("/home/skynet/Downloads/hawk eye/data/ball not/"),
        output_ball_dir=Path("/home/skynet/Downloads/hawk eye/data/_ball/"),
        output_not_ball_dir=Path("/home/skynet/Downloads/hawk eye/data/_ball not/"),
        frames_dir=Path("/home/skynet/Downloads/hawk eye/data/frames/"),
        candidates_dir=Path("/home/skynet/Downloads/hawk eye/data/motion_candidates/"),
        diff_frames_dir=Path("/home/skynet/Downloads/hawk eye/data/diff_frames/"),
        expansion_factor=2.0,
        output_size=(64, 64),
        padding_mode="constant",
        preview_enabled=True,
        preview_ext=".jpg",
        strict_filename_parsing=True,
        flatten_output=False,
        npz_dir_name="npz",
        preview_dir_name="jpg",
    )
    builder = AnnotatedCandidateDatasetBuilder(config)
    builder.run()


if __name__ == "__main__":
    main()
