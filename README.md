# Vehicle Number Plate Detection Backend

A simple FastAPI backend that accepts a vehicle image and returns the detected
number plate text, using EasyOCR.

## Endpoints

- `GET /` — health check
- `POST /detect-plate` — send an image file (form field name: `file`), get back:
  ```json
  {
    "plate_found": true,
    "plate_text": "KA01AB1234",
    "confidence": 0.97,
    "box": { "x": 167, "y": 173, "width": 228, "height": 42 },
    "other_candidates": []
  }
  ```

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs to test it interactively (FastAPI's
built-in Swagger UI — click "Try it out" on `/detect-plate` and upload a photo).

## Deploy on Railway (via GitHub)

See the step-by-step guide from Claude for the full walkthrough. Summary:

1. Push this folder to a new GitHub repository.
2. On Railway, create a New Project → Deploy from GitHub repo → select the repo.
3. Railway auto-detects Python, installs `requirements.txt`, and starts the
   app using the `Procfile` (`uvicorn main:app --host 0.0.0.0 --port $PORT`).
4. Once deployed, Railway gives you a public URL like
   `https://your-app.up.railway.app`. Test it by uploading an image to
   `https://your-app.up.railway.app/detect-plate`.

## Notes

- The first request after deploy will be slow (EasyOCR downloads its model,
  ~65MB, the first time it runs). Subsequent requests are much faster.
- Accuracy depends heavily on photo quality — front-facing, well-lit, and not
  too far away works best.
- `PLATE_PATTERN` in `main.py` filters OCR results to plate-like strings
  (4-12 letters/digits). If your plates have spaces or hyphens (e.g. some
  formats), you may want to loosen this pattern.
