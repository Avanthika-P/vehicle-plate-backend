"""
Vehicle Number Plate Detection Backend (EasyOCR version)
----------------------------------------------------------
Upload a vehicle image -> EasyOCR detects all text regions in the photo ->
regions are clustered (to handle 2-line plates) -> filtered against the
standard Indian plate format -> best match returned.

Uses EasyOCR (deep-learning based) rather than classical OCR, since it
handles real-world photos (compression artifacts, blur, small text,
cluttered backgrounds) far more reliably. This needs more memory than
Tesseract, so requires a paid/upgraded hosting tier rather than a free one.

Endpoints:
  GET  /                 -> health check
  POST /detect-plate      -> upload an image, get back plate text
"""

import io
import os
import re

import cv2
import numpy as np
import easyocr
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

# Loaded once at startup. Models are pre-downloaded during the Docker
# build (see Dockerfile), so this doesn't re-download at container start.
reader = easyocr.Reader(["en"], gpu=False)

MAX_DIMENSION = 1600

# Standard Indian plate format: 2 state letters, 1-2 digit RTO code,
# 1-2 series letters, 4 digit number. e.g. MH12KR1145, TN05BT5754.
STRICT_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$")


def is_loose_plate_like(s: str) -> bool:
    """A plausible-but-unverified plate: right length, and a genuine mix
    of letters and digits (excludes pure-text banner words and pure-digit
    phone numbers, which otherwise cause false positives)."""
    if not (4 <= len(s) <= 11):
        return False
    n_digits = sum(ch.isdigit() for ch in s)
    n_letters = sum(ch.isalpha() for ch in s)
    return n_digits >= 1 and n_letters >= 2


def read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    width, height = image.size
    longest_side = max(width, height)
    if longest_side > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest_side
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def box_bounds(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def cluster_results(results):
    """Group nearby text detections together, so a 2-line plate (e.g.
    'MH12K' directly above 'R1145') gets read as one combined string.
    Groups by: similar horizontal position + close vertical gap."""
    items = []
    for bbox, text, conf in results:
        x1, y1, x2, y2 = box_bounds(bbox)
        items.append(
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "text": text, "conf": conf}
        )

    # Sort top-to-bottom so we can greedily chain nearby lines.
    items.sort(key=lambda i: i["y1"])

    used = [False] * len(items)
    groups = []
    for i, item in enumerate(items):
        if used[i]:
            continue
        group = [item]
        used[i] = True
        height = item["y2"] - item["y1"]
        cur = item
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            other = items[j]
            horizontal_overlap = min(cur["x2"], other["x2"]) - max(cur["x1"], other["x1"])
            min_width = min(cur["x2"] - cur["x1"], other["x2"] - other["x1"])
            vertical_gap = other["y1"] - cur["y2"]
            if (
                min_width > 0
                and horizontal_overlap / min_width > 0.3
                and 0 <= vertical_gap <= height * 1.2
            ):
                group.append(other)
                used[j] = True
                cur = other
                height = other["y2"] - other["y1"]
        groups.append(group)

    combined = []
    for group in groups:
        text = "".join(g["text"] for g in group)
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        conf = sum(g["conf"] for g in group) / len(group)
        x1 = min(g["x1"] for g in group)
        y1 = min(g["y1"] for g in group)
        x2 = max(g["x2"] for g in group)
        y2 = max(g["y2"] for g in group)
        combined.append(
            {
                "text": cleaned,
                "confidence": conf,
                "box": {"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)},
            }
        )
    return combined


def find_plate_candidates(img: np.ndarray):
    results = reader.readtext(img)
    combined = cluster_results(results)

    strict_matches = [c for c in combined if STRICT_PLATE_PATTERN.match(c["text"])]
    if strict_matches:
        strict_matches.sort(key=lambda c: c["confidence"], reverse=True)
        return strict_matches, "strict"

    loose_matches = [c for c in combined if is_loose_plate_like(c["text"])]
    loose_matches.sort(key=lambda c: c["confidence"], reverse=True)
    return loose_matches, "loose"


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
        candidates, match_type = find_plate_candidates(img)
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
        "match_confidence": match_type,
        "ocr_confidence": round(best["confidence"], 3),
        "box": best["box"],
        "other_candidates": candidates[1:5],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
