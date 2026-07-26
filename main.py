"""
Vehicle Number Plate Detection Backend
---------------------------------------
Upload a vehicle image -> locates candidate plate regions using classical
computer vision (edge detection + contour shape filtering) -> reads the
text of each candidate with Tesseract OCR -> returns the best match.

Uses Tesseract (lightweight, low memory) rather than a deep-learning OCR
engine, since this needs to run comfortably within default cloud memory
limits (e.g. Railway's free/starter tier).

Endpoints:
  GET  /                 -> health check
  POST /detect-plate      -> upload an image, get back plate text + coordinates
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

MAX_DIMENSION = 1600  # cap the longest side to keep memory/time usage low

# A plate-like string: mostly letters/digits, length 4-12 (covers most
# country formats loosely). Tune this for your region if needed.
PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,12}$")


def read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    width, height = image.size
    longest_side = max(width, height)
    if longest_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest_side
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def find_candidate_regions(img: np.ndarray):
    """Locate rectangular regions in the image that look like they could
    be a license plate, based on aspect ratio and size. Returns a list of
    (x, y, w, h) boxes, most-likely-first."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = gray.shape[:2]
    img_area = img_h * img_w

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        area = w * h
        aspect_ratio = w / h

        # A number plate is usually a wide rectangle, not tiny, not the
        # whole image.
        if (
            2.0 <= aspect_ratio <= 6.0
            and 0.0015 * img_area <= area <= 0.2 * img_area
            and w > 60
        ):
            boxes.append((x, y, w, h))

    # Larger candidates first tend to be more reliable.
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:15]  # cap how many we OCR, for speed


def ocr_region(img: np.ndarray, box) -> str:
    x, y, w, h = box
    pad = int(0.1 * h)
    y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
    crop = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def find_plate_candidates(img: np.ndarray):
    """Find candidate plate regions, OCR each one, and return the ones
    whose text looks like a plate -- sorted by how plate-like the region's
    shape was (a reasonable proxy for confidence with this approach)."""
    boxes = find_candidate_regions(img)

    candidates = []
    for box in boxes:
        text = ocr_region(img, box)
        if PLATE_PATTERN.match(text):
            x, y, w, h = box
            candidates.append(
                {
                    "text": text,
                    "confidence": None,  # classical CV approach has no
                                          # calibrated confidence score
                    "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                }
            )

    return candidates


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Vehicle Plate Detection API is running"}


@app.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    img = read_image_from_upload(file_bytes)
    try:
        candidates = find_plate_candidates(img)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {exc}")

    if not candidates:
        return {
            "plate_found": False,
            "message": "No plate detected. Try a clearer, more front-facing photo.",
        }

    best = candidates[0]
    return {
        "plate_found": True,
        "plate_text": best["text"],
        "box": best["box"],
        "other_candidates": candidates[1:],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
