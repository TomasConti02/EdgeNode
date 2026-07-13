FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.28.0 \
    httpx==0.27.0 \
    python-multipart==0.0.9

COPY RestAPI.py .

EXPOSE 5000

CMD ["uvicorn", "RestAPI:app", "--host", "0.0.0.0", "--port", "5000"]
