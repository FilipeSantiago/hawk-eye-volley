from pathlib import Path
import json

import cv2


FRAMES_DIR = Path(
    "/home/bugslayer/Downloads/volley video footage/data/frames/"
    "videoplayback-00.06.05.191-00.11.14.198"
)
ANNOTATED_DIR = Path("/home/bugslayer/Downloads/volley video footage/data/annotated_frames")
CANDIDATES_DIR = Path("/home/bugslayer/Downloads/volley video footage/data/candidates")
PERSON_ANN_DIR = Path("/home/bugslayer/Downloads/volley video footage/data/person_annotations")
DIFF_THRESHOLD = 12
MIN_MOVING_AREA = 1


def iter_frames(frames_dir: Path) -> list[Path]:
    return sorted(p for p in frames_dir.glob("*.jpg") if p.is_file())


def motion_mask(prev_frame, curr_frame, threshold: int):
    diff = cv2.absdiff(curr_frame, prev_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def load_person_boxes(annotation_path: Path):
    if not annotation_path.is_file():
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


def mask_out_boxes(mask, boxes):
    h, w = mask.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 0
    return mask


def extract_boxes(mask):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < MIN_MOVING_AREA:
            continue
        boxes.append((int(x), int(y), int(w), int(h), int(area)))
    return boxes


def draw_boxes(image, boxes, color=(0, 255, 255), thickness=2):
    vis = image.copy()
    for x, y, w, h, _ in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    return vis


def write_candidates_json(path: Path, frame_name: str, index: int, boxes):
    payload = {
        "frame": frame_name,
        "index": index,
        "num_candidates": len(boxes),
        "candidates": [
            {"bbox": [x, y, w, h], "area": area} for x, y, w, h, area in boxes
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    frames_dir = FRAMES_DIR
    annotated_dir = ANNOTATED_DIR
    candidates_dir = CANDIDATES_DIR
    person_ann_dir = PERSON_ANN_DIR
    annotated_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = iter_frames(frames_dir)
    if len(frame_paths) < 2:
        raise FileNotFoundError(f"Need at least 2 frames in: {frames_dir}")

    processed = 0
    for idx in range(1, len(frame_paths)):
        prev_path = frame_paths[idx - 1]
        curr_path = frame_paths[idx]
        prev = cv2.imread(str(prev_path))
        curr = cv2.imread(str(curr_path))
        if prev is None or curr is None:
            continue

        mask = motion_mask(prev, curr, DIFF_THRESHOLD)
        person_boxes = load_person_boxes(person_ann_dir / f"{curr_path.stem}.json")
        if person_boxes:
            mask = mask_out_boxes(mask, person_boxes)
        boxes = extract_boxes(mask)
        vis = draw_boxes(curr, boxes)
        cv2.imwrite(str(annotated_dir / curr_path.name), vis)
        write_candidates_json(candidates_dir / f"{curr_path.stem}.json", curr_path.name, idx, boxes)
        processed += 1

    print(f"Processed {processed} frames into {annotated_dir}")


if __name__ == "__main__":
    main()
