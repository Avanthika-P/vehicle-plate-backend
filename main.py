"""
Vehicle Number Plate Detection Backend
---------------------------------------
Upload a vehicle image -> reads the plate text with Tesseract OCR.

Works best when the photo is reasonably zoomed in on the plate itself
(fills most of the frame, straight-on, well lit) -- this keeps the
approach lightweight enough to run on low-memory hosting (e.g. Railway's
free/starter tier) without needing a deep-learning OCR engine.

Strategy:
  1. Try reading the image directly, assuming it's already a zoomed-in
     photo of just the plate (the recommended way to use this API).
  2. If that doesn't yield a plate-like result, fall back to scanning the
     image for plate-shaped regions (by color + shape) and OCR those.

Endpoints:
  GET  /                 -> health check
  POST /detect-plate      -> upload an image, get back plate text
"""

import io
import os
import re

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Vehicle Plate Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_DIMENSION = 1200  # cap the longest side to keep memory/time usage low

# Matches standard Indian plate format: 2 state letters, 1-2 digit RTO
# code, 1-2 series letters, 4 digit number. e.g. MH12KR1145, TN05BT5754.
# OCR text is read line-by-line and concatenated before matching, so this
# works for both 1-line and 2-line plate layouts.
STRICT_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$")
# Looser fallback for other formats / partial reads.
LOOSE_PLATE_PATTERN = re.compile(r"^[A-Z0-9]{6,11}$")

TESSERACT_CONFIG = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    width, height = image.size
    longest_side = max(width, height)
    if longest_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest_side
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def clean_text(raw: str) -> str:
    """Join OCR lines and strip everything but letters/digits, so a
    2-line plate like 'MH12K\\nR1145' becomes 'MH12KR1145'."""
    joined = "".join(raw.split())
    return re.sub(r"[^A-Z0-9]", "", joined.upper())


def ocr_image(img: np.ndarray) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # psm 6: "assume a uniform block of text" -- handles both 1-line and
    # 2-line plates well, unlike psm 7 (single line only).
    text = pytesseract.image_to_string(thresh, config=f"--psm 6 {TESSERACT_CONFIG}")
    return clean_text(text)


def find_region_candidates(img: np.ndarray):
    """Fallback: look for plate-shaped regions within a larger photo
    (color + shape based, no deep learning) and OCR each one."""
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_full = cv2.Canny(gray_full, 50, 150)

    all_boxes = []

    # Yellow commercial plates (India)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_boxes.extend(cv2.boundingRect(c) for c in contours)

    # General edges (white/other plates)
    gray = cv2.bilateralFilter(gray_full, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
    contours2, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_boxes.extend(cv2.boundingRect(c) for c in contours2)

    scored = []
    for x, y, w, h in all_boxes:
        if h == 0 or w < 60:
            continue
        ar = w / h
        area_frac = (w * h) / img_area
        if not (1.5 <= ar <= 6.0 and 0.001 <= area_frac <= 0.05):
            continue
        region_edges = edges_full[y : y + h, x : x + w]
        edge_density = cv2.countNonZero(region_edges) / (w * h)
        scored.append((edge_density, x, y, w, h))

    scored.sort(key=lambda s: s[0], reverse=True)
    return [(x, y, w, h) for _, x, y, w, h in scored[:20]]


def detect_plate(img: np.ndarray):
    h, w = img.shape[:2]

    # 1. Assume it's already a zoomed-in photo of the plate.
    whole_image_text = ocr_image(img)
    if STRICT_PLATE_PATTERN.match(whole_image_text):
        return {
            "text": whole_image_text,
            "box": {"x": 0, "y": 0, "width": w, "height": h},
            "match": "strict",
        }

    best_loose = None
    if LOOSE_PLATE_PATTERN.match(whole_image_text):
        best_loose = {
            "text": whole_image_text,
            "box": {"x": 0, "y": 0, "width": w, "height": h},
            "match": "loose",
        }

    # 2. Fall back to scanning for plate-shaped regions in a larger photo.
    for x, y, rw, rh in find_region_candidates(img):
        pad = int(0.08 * rh)
        y1, y2 = max(0, y - pad), min(img.shape[0], y + rh + pad)
        x1, x2 = max(0, x - pad), min(img.shape[1], x + rw + pad)
        crop = img[y1:y2, x1:x2]
        crop_text = ocr_image(crop)

        if STRICT_PLATE_PATTERN.match(crop_text):
            return {
                "text": crop_text,
                "box": {"x": int(x), "y": int(y), "width": int(rw), "height": int(rh)},
                "match": "strict",
            }
        if best_loose is None and LOOSE_PLATE_PATTERN.match(crop_text):
            best_loose = {
                "text": crop_text,
                "box": {"x": int(x), "y": int(y), "width": int(rw), "height": int(rh)},
                "match": "loose",
            }

    return best_loose


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Vehicle Plate Detection API is running"}


@app.post("/detect-plate")
async def detect_plate_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    img = read_image_from_upload(file_bytes)
    try:
        result = detect_plate(img)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {exc}")

    if result is None:
        return {
            "plate_found": False,
            "message": (
                "No plate detected. For best results, take a photo zoomed in "
                "on just the plate -- straight-on, well lit, filling most of "
                "the frame."
            ),
        }

    return {
        "plate_found": True,
        "plate_text": result["text"],
        "match_confidence": result["match"],  # "strict" = matched standard
                                                # Indian plate format;
                                                # "loose" = plausible but
                                                # unverified format
        "box": result["box"],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
