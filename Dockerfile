FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, Tesseract OCR, and Poppler (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (bert_model/ is included via COPY)
COPY . .

# Expose port
EXPOSE 5000

# Run with Gunicorn — 1 worker because the BERT model is memory-heavy. We bind to $PORT injected by Railway.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 app:flask_app
