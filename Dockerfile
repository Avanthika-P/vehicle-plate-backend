FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR's models during the BUILD step (which has a long
# timeout and isn't user-facing), so the running container starts
# instantly instead of downloading ~65MB+ of models on first request.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY . .

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
