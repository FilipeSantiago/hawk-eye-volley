import json
from pathlib import Path

import cv2

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar
    def tqdm(iterable, total=None, desc=None):
        return iterable

FRAME_DIR = Path(
    "/home/bugslayer/Downloads/volley video footage/data/frames/"
    "videoplayback-00.06.05.191-00.11.14.198/"
)
DIFF_DIR = Path("/home/bugslayer/Downloads/volley video footage/data/diff_frames/")
CANDIDATE_DIR = Path("/home/bugslayer/Downloads/volley video footage/data/candidates/")
PERSON_ANN_DIR = Path("/home/bugslayer/Downloads/volley video footage/data/person_annotations")


def iter_frame_files(frame_dir: Path = FRAME_DIR):
    """Yield files in frame_dir, sorted by name."""
    for path in sorted(frame_dir.iterdir()):
        if path.is_file():
            yield path


def iter_correlated_files(
    frame_dir: Path = FRAME_DIR,
    diff_dir: Path = DIFF_DIR,
    candidate_dir: Path = CANDIDATE_DIR,
):
    """Yield tuples of (frame_path, diff_frame_path|None, candidate_path|None)."""
    for frame_path in iter_frame_files(frame_dir):
        stem = frame_path.stem  # frame_000001
        if not stem.startswith("frame_"):
            continue
        index = stem.split("_", 1)[1]
        diff_path = diff_dir / f"diff_frame_{index}.png"
        candidate_path = candidate_dir / f"frame_{index}.json"
        yield (
            frame_path,
            diff_path if diff_path.is_file() else None,
            candidate_path if candidate_path.is_file() else None,
        )


def load_candidates(candidate_path: Path):
    if candidate_path is None or not candidate_path.is_file():
        return []
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    return candidates if isinstance(candidates, list) else []


def draw_candidate_boxes(image, candidates, color=(0, 255, 255), thickness=2):
    vis = image.copy()
    img_h, img_w = vis.shape[:2]
    for candidate in candidates:
        bbox = candidate.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = [int(round(v)) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w - 1, x + w)
        y2 = min(img_h - 1, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def load_person_boxes(annotation_path: Path):
    if annotation_path is None or not annotation_path.is_file():
        return []
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    detections = payload.get("detections", [])
    boxes = []
    for det in detections:
        if det.get("class_id") != 0:
            continue
        xyxy = det.get("bbox_xyxy")
        if not xyxy or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def draw_person_boxes(image, boxes, color=(0, 200, 0), thickness=2):
    vis = image.copy()
    img_h, img_w = vis.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def iter_correlated_images_with_boxes(
    frame_dir: Path = FRAME_DIR,
    diff_dir: Path = DIFF_DIR,
    candidate_dir: Path = CANDIDATE_DIR,
    person_ann_dir: Path = PERSON_ANN_DIR,
    max_candidates=None,
    log_missing: bool = True,
):
    for frame_path, diff_path, candidate_path in iter_correlated_files(
        frame_dir=frame_dir,
        diff_dir=diff_dir,
        candidate_dir=candidate_dir,
    ):
        if diff_path is None:
            if log_missing:
                print(f"Skip {frame_path.name}: missing diff image")
            continue
        original = cv2.imread(str(frame_path))
        diff = cv2.imread(str(diff_path))
        if original is None or diff is None:
            if log_missing:
                print(f"Skip {frame_path.name}: failed to read image(s)")
            continue
        if candidate_path is None or not candidate_path.is_file():
            if log_missing:
                print(f"Info {frame_path.name}: missing candidates JSON")
            candidates = []
        else:
            candidates = load_candidates(candidate_path)
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        person_ann_path = person_ann_dir / f"{frame_path.stem}.json"
        if not person_ann_path.is_file():
            if log_missing:
                print(f"Info {frame_path.name}: missing person annotations")
            person_boxes = []
        else:
            person_boxes = load_person_boxes(person_ann_path)
        original_boxed = draw_candidate_boxes(original, candidates)
        diff_boxed = draw_candidate_boxes(diff, candidates)
        if person_boxes:
            original_boxed = draw_person_boxes(original_boxed, person_boxes)
            diff_boxed = draw_person_boxes(diff_boxed, person_boxes)
        yield (original, original_boxed, diff, diff_boxed)


def _ensure_bgr(image, size):
    if image is None:
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if size is not None:
        width, height = size
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def compose_grid(original, original_boxed, diff, diff_boxed):
    base_h, base_w = original.shape[:2]
    size = (base_w, base_h)
    original = _ensure_bgr(original, size)
    original_boxed = _ensure_bgr(original_boxed, size)
    diff = _ensure_bgr(diff, size)
    diff_boxed = _ensure_bgr(diff_boxed, size)
    top = cv2.hconcat([original, original_boxed])
    bottom = cv2.hconcat([diff, diff_boxed])
    return cv2.vconcat([top, bottom])


def write_grid_video(
    output_path: Path,
    frame_dir: Path = FRAME_DIR,
    diff_dir: Path = DIFF_DIR,
    candidate_dir: Path = CANDIDATE_DIR,
    person_ann_dir: Path = PERSON_ANN_DIR,
    fps: int = 30,
    max_candidates=None,
    codec: str = "mp4v",
    log_missing: bool = True,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = sum(
        1
        for _, diff_path, _ in iter_correlated_files(
            frame_dir=frame_dir,
            diff_dir=diff_dir,
            candidate_dir=candidate_dir,
        )
        if diff_path is not None and diff_path.is_file()
    )

    writer = None
    try:
        frames = iter_correlated_images_with_boxes(
            frame_dir=frame_dir,
            diff_dir=diff_dir,
            candidate_dir=candidate_dir,
            person_ann_dir=person_ann_dir,
            max_candidates=max_candidates,
            log_missing=log_missing,
        )
        for original, original_boxed, diff, diff_boxed in tqdm(
            frames, total=total, desc="Writing video"
        ):
            grid = compose_grid(original, original_boxed, diff, diff_boxed)
            if writer is None:
                height, width = grid.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {output_path}")
            writer.write(grid)
    finally:
        if writer is not None:
            writer.release()


write_grid_video("/home/bugslayer/Downloads/volley video footage/data/test.mp4")
