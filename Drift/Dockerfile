# Usa un'immagine ufficiale Python leggera
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
# Install python-multipart for FastAPI Form/File handling
RUN pip install --no-cache-dir fastapi uvicorn numpy scikit-learn requests redis python-multipart

COPY driftv3.py .
COPY centroids.npy inv_cov_matrix.npy .

CMD ["uvicorn", "driftv3:app", "--host", "0.0.0.0", "--port", "8080"]
