from __future__ import annotations

import re

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FILENAME_PATTERN = re.compile(r"^(frame_\d+)_candidate_(\d+)$")
TOKEN_SANITIZER_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
PADDING_MODE_TO_CV2 = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT_101,
}

