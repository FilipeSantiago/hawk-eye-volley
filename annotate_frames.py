from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COCO_PERSON = 0
COCO_SPORTS_BALL = 32


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO on images and save detections as JSON."
    )
    parser.add_argument(
        "--input-dir",
        default=(
            "/home/bugslayer/Downloads/volley video footage/data/frames/videoplayback-00.06.05.191-00.11.14.198/"
        ),
        help="Directory containing extracted frames.",
    )
    parser.add_argument(
        "--annotations-dir",
        default="/home/bugslayer/Downloads/volley video footage/data/person_annotations",
        help="Directory to write annotation JSON files.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Path to a YOLO model.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=None,
        help="Inference image size: single int or H W. Overrides --imgsz-native.",
    )
    parser.add_argument(
        "--imgsz-native",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use native image size for inference (avoid downscaling).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Base confidence threshold (used if per-class thresholds are not set).",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=None,
        help="Confidence threshold for person class. Defaults to --conf.",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=None,
        help="Confidence threshold for sports ball class. Defaults to --conf.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, e.g. 'cuda:0' or 'cpu'. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of images per inference batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    annotations_dir = Path(args.annotations_dir).expanduser()
    annotations_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    base_conf = args.conf
    person_conf = args.person_conf if args.person_conf is not None else base_conf
    ball_conf = args.ball_conf if args.ball_conf is not None else base_conf
    infer_conf = min(person_conf, ball_conf)

    model = YOLO(args.model)
    images = iter_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    imgsz = None
    if args.imgsz:
        if len(args.imgsz) == 1:
            imgsz = int(args.imgsz[0])
        elif len(args.imgsz) == 2:
            imgsz = (int(args.imgsz[0]), int(args.imgsz[1]))
        else:
            raise ValueError("--imgsz must be a single int or two ints (H W).")
    elif args.imgsz_native:
        sample = cv2.imread(str(images[0]))
        if sample is None:
            raise RuntimeError(f"Failed to read image: {images[0]}")
        height, width = sample.shape[:2]
        imgsz = (height, width)

    print(f"Input dir: {input_dir}")
    print(f"Annotations dir: {annotations_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Images: {len(images)}")
    print(f"Batch size: {max(1, args.batch_size)}")
    print(f"Image size: {imgsz if imgsz is not None else 'default'}")
    print(f"Confidence (person): {person_conf}")
    print(f"Confidence (sports ball): {ball_conf}")

    batch_size = max(1, args.batch_size)
    for start in range(0, len(images), batch_size):
        batch_paths = images[start : start + batch_size]
        batch_sources = [str(p) for p in batch_paths]
        batch_index = start // batch_size + 1
        total_batches = (len(images) + batch_size - 1) // batch_size
        print(
            f"Batch {batch_index}/{total_batches}: "
            f"{start + 1}-{start + len(batch_paths)}"
        )
        predict_kwargs = {
            "source": batch_sources,
            "classes": [COCO_PERSON, COCO_SPORTS_BALL],
            "conf": infer_conf,
            "device": device,
            "batch": batch_size,
            "verbose": False,
        }
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        results = model.predict(**predict_kwargs)

        for img_path, result in zip(batch_paths, results):
            detections: list[dict[str, object]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    cls_id = int(result.boxes.cls[i].item())
                    conf = float(result.boxes.conf[i].item())
                    if cls_id == COCO_PERSON and conf < person_conf:
                        continue
                    if cls_id == COCO_SPORTS_BALL and conf < ball_conf:
                        continue
                    xyxy = [float(v) for v in result.boxes.xyxy[i].tolist()]
                    detections.append(
                        {
                            "class_id": cls_id,
                            "class_name": result.names.get(cls_id, str(cls_id)),
                            "confidence": conf,
                            "bbox_xyxy": xyxy,
                        }
                    )

            ann_payload = {
                "image": str(img_path),
                "width": int(result.orig_shape[1]),
                "height": int(result.orig_shape[0]),
                "detections": detections,
            }

            ann_path = annotations_dir / f"{img_path.stem}.json"
            ann_path.write_text(json.dumps(ann_payload, indent=2), encoding="utf-8")
            print(f"Saved: {ann_path.name} | detections: {len(detections)}")

    print(f"Annotations: {annotations_dir}")


if __name__ == "__main__":
    main()
