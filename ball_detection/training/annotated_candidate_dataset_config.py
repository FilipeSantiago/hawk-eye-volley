from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ball_detection.utils.candidate_dataset_constants import PADDING_MODE_TO_CV2


@dataclass
class AnnotatedCandidateDatasetConfig:
    annotated_ball_dir: Path | str = Path("/home/skynet/Downloads/hawk eye/data/ball/")
    annotated_not_ball_dir: Path | str = Path("/home/skynet/Downloads/hawk eye/data/ball not/")
    output_ball_dir: Path | str | None = None
    output_not_ball_dir: Path | str | None = None
    frames_dir: Path | str = Path("/home/skynet/Downloads/hawk eye/data/frames/")
    candidates_dir: Path | str = Path("/home/skynet/Downloads/hawk eye/data/motion_candidates/")
    diff_frames_dir: Path | str = Path("/home/skynet/Downloads/hawk eye/data/diff_frames/")
    expansion_factor: float = 2.0
    output_size: tuple[int, int] = (64, 64)  # (width, height)
    padding_mode: str = "constant"
    preview_enabled: bool = True
    preview_ext: str = ".jpg"
    strict_filename_parsing: bool = True
    flatten_output: bool = False
    npz_dir_name: str = "npz"
    preview_dir_name: str = "jpg"

    def __post_init__(self) -> None:
        self.annotated_ball_dir = Path(self.annotated_ball_dir).expanduser()
        self.annotated_not_ball_dir = Path(self.annotated_not_ball_dir).expanduser()
        self.frames_dir = Path(self.frames_dir).expanduser()
        self.candidates_dir = Path(self.candidates_dir).expanduser()
        self.diff_frames_dir = Path(self.diff_frames_dir).expanduser()

        if self.output_ball_dir is None:
            self.output_ball_dir = self.annotated_ball_dir.parent / f"_{self.annotated_ball_dir.name}"
        self.output_ball_dir = Path(self.output_ball_dir).expanduser()

        if self.output_not_ball_dir is None:
            self.output_not_ball_dir = (
                self.annotated_not_ball_dir.parent / f"_{self.annotated_not_ball_dir.name}"
            )
        self.output_not_ball_dir = Path(self.output_not_ball_dir).expanduser()

        if not self.preview_ext.startswith("."):
            self.preview_ext = f".{self.preview_ext}"

        self.npz_dir_name = self.npz_dir_name.strip()
        self.preview_dir_name = self.preview_dir_name.strip()
        if not self.npz_dir_name:
            raise ValueError("npz_dir_name must not be empty")
        if not self.preview_dir_name:
            raise ValueError("preview_dir_name must not be empty")

        output_w, output_h = self.output_size
        self.output_size = (max(1, int(output_w)), max(1, int(output_h)))
        self.expansion_factor = float(self.expansion_factor)
        if self.expansion_factor <= 0:
            raise ValueError("expansion_factor must be > 0")

        mode = self.padding_mode.lower().strip()
        if mode not in PADDING_MODE_TO_CV2:
            raise ValueError(
                f"Unsupported padding_mode={self.padding_mode!r}. "
                f"Use one of: {', '.join(sorted(PADDING_MODE_TO_CV2))}"
            )
        self.padding_mode = mode
