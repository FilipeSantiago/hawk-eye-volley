from __future__ import annotations

from pathlib import Path

from ball_detection.motion_detector import MotionDetector
from utils.video import video_to_frames


def main() -> None:
    output_dir = Path("/home/skynet/Downloads/hawk eye/data/frames/")
    video = "/home/skynet/Downloads/hawk eye/data/videoplayback.1773931843922.publer.com-00.00.00.000-00.10.03.976-00.06.08.875-00.10.04.033.mp4"

    output_dir = Path(output_dir).expanduser()
    saved_frames, frames_dir = video_to_frames(video, output_dir)
    print(f"Saved {saved_frames} frame(s) to: {frames_dir}")

    data_dir = output_dir.parent
    video_name = frames_dir.name
    annotated_dir = data_dir / "motion_annotated" / video_name
    candidates_dir = data_dir / "motion_candidates" / video_name
    crops_dir = data_dir / "motion_candidate_crops" / video_name

    detector = MotionDetector(
        frames_dir=frames_dir,
        annotated_dir=annotated_dir,
        candidates_dir=candidates_dir,
        min_moving_area=30,
        max_moving_area=350,
        diff_threshold=12,
    )
    processed = detector.run_motion_detector()
    exported_crops = detector.export_cropped_candidates(crops_dir)

    print(f"Processed {processed} frame pair(s) for motion detection")
    print(f"Candidates JSON saved in: {candidates_dir}")
    print(f"Exported {exported_crops} cropped candidates to: {crops_dir}")


if __name__ == "__main__":
    main()
