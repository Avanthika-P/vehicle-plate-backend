# Vehicle Number Plate Detection Backend

A lightweight FastAPI backend that reads Indian vehicle plates from photos,
using Tesseract OCR (no deep learning / GPU needed -- runs fine on
low-memory hosting like Railway's free tier).

## For best accuracy, take the photo like this:
- **Zoom in on the plate itself** -- it should fill most of the frame
- Straight-on angle (not a steep side angle)
- Good lighting, avoid glare/reflection on the plate
- Avoid heavy compression (e.g. sending an original photo rather than a
  re-shared/forwarded WhatsApp image, which recompresses it)

The API also tries to find the plate automatically in a wider photo (e.g.
the whole back of the vehicle), but this is best-effort with classical
computer vision and works less reliably on busy/cluttered scenes -- for
guaranteed accuracy, crop to the plate before uploading.

## Endpoints

- `GET /` -- health check
- `POST /detect-plate` -- send an image file (form field name: `file`), get back:
  ```json
  {
    "plate_found": true,
    "plate_text": "MH12KR1145",
    "match_confidence": "strict",
    "box": { "x": 0, "y": 0, "width": 400, "height": 200 }
  }
  ```
  `match_confidence` is `"strict"` when the text matches standard Indian
  plate format (2 letters + 1-2 digits + 1-2 letters + 4 digits), or
  `"loose"` when it's a plausible but unverified read.

## Run locally

```bash
pip install -r requirements.txt
# You also need tesseract-ocr installed on your system:
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki
#   Mac:     brew install tesseract
#   Linux:   sudo apt-get install tesseract-ocr
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs to test interactively.

## Deploy on Railway (via GitHub)

1. Push this folder to a GitHub repository
2. On Railway: New Project → Deploy from GitHub repo → select the repo
3. Railway detects the `Dockerfile` and builds it (this installs
   `tesseract-ocr` inside the image -- see `Dockerfile`)
4. Settings → Networking → Generate Domain to get a public URL
5. Test at `https://your-url/docs`

## Notes on accuracy

This uses classical computer vision (color + shape based region detection)
rather than a trained plate-detector model, to keep memory usage low enough
for free-tier hosting. This means:
- Zoomed-in, well-lit plate photos: works well
- Full vehicle photos from a distance/angle, especially with background
  clutter (signage, decals, etc.): less reliable -- OCR may pick up other
  text, or miss the plate

If you need reliable detection on full, uncropped vehicle photos, the more
robust option is a deep-learning-based OCR engine (e.g. EasyOCR) or a
trained plate-detector model, which needs more memory than most free
hosting tiers provide.
