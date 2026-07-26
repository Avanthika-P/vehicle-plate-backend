"""
Vehicle Number Plate Detection Backend
---------------------------------------
Upload a vehicle image -> detects the number plate region -> reads the text.

Endpoints:
  GET  /                 -> health check
  POST /detect-plate      -> upload an image, get back plate text + coordinates
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

# Allow requests from any frontend (adjust in production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# EasyOCR reader is loaded once at startup (downloads its model the first
# time it runs). This does both text DETECTION and RECOGNITION, so we don't
# need a separate plate-localization step.
reader = easyocr.Reader(["en"], gpu=False)

# A plate-like string: mostly letters/digits, length 4-12 (covers most
# country formats loosely). Tune this for your region if needed.
PLATE_PATTERN = re.compile(r"^[A-Z0-9]{4,12}$")


def read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def find_plate_candidates(img: np.ndarray):
    """Run OCR over the whole image and return results that look like a
    plate, sorted by confidence (highest first)."""
    results = reader.readtext(img)

    candidates = []
    for bbox, text, confidence in results:
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        if PLATE_PATTERN.match(cleaned):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            candidates.append(
                {
                    "text": cleaned,
                    "confidence": float(confidence),
                    "box": {"x": x, "y": y, "width": w, "height": h},
                }
            )

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
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
    candidates = find_plate_candidates(img)

    if not candidates:
        return {
            "plate_found": False,
            "message": "No plate detected. Try a clearer, more front-facing photo.",
        }

    best = candidates[0]
    return {
        "plate_found": True,
        "plate_text": best["text"],
        "confidence": round(best["confidence"], 3),
        "box": best["box"],
        "other_candidates": candidates[1:],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
