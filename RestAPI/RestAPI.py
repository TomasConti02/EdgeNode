import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import List
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RestAPI")
# istio routing gataway to reach the inferences services
ISTIO_GATEWAY = os.getenv("ISTIO_GATEWAY", "http://istio-ingressgateway.istio-system.svc.cluster.local")
IMAGE_SIZE = 224 * 224 * 3 #size of the raw images in bytes 

httpx_client: httpx.AsyncClient = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global httpx_client
    httpx_client = httpx.AsyncClient(timeout=30.0)
    logger.info("HTTP client ready")
    yield
    await httpx_client.aclose() #clean up the network resource 

app = FastAPI(lifespan=lifespan)

##########################################################################################################################################
async def forward(img: bytes, model: str, content_type: str, image_key: str):
    try:
        headers = {
            "Host": f"{model}.default.example.com", # this host information is used by istio gateways for internal routing to the target
            "Content-Type": content_type, #type of the payload 
            "X-Image-Key": image_key  #ood redis key 
        } 
        response = await httpx_client.post( # # I/O netwrok no blocking operation . launch the http rest post request to the inference service and collect the response
            f"{ISTIO_GATEWAY}/v1/models/{model}:predict", # the call is directed to the istio gataway, asking for model predict endpoint 
            headers=headers,
            content=img, #raw images data, not overhead and other pre processing ( transformer operation)
        )
        response.raise_for_status()
        result = response.json()
        logger.info("%s -> %d (Redis Key: %s)", model, response.status_code, image_key)
        return {"model": model, "predicted_class": result.get("predicted_class"), "image_key": image_key}
    
    except httpx.HTTPStatusError as exc:
        logger.error("%s -> %d (Redis Key: %s)", model, exc.response.status_code, image_key)
        return {"model": model, "error": exc.response.text, "image_key": image_key}
    except Exception as exc:
        logger.exception("%s failed (Redis Key: %s)", model, image_key)
        return {"model": model, "error": str(exc), "image_key": image_key}

##########################################################################################################################################
async def store_image_to_detector( img: bytes,  filename: str,  content_type: str,  model: str,  image_key: str ):
    detector_host = f"ood-detector-{model}.default.example.com"
    # pack metadata into http header
    headers = {
        "Host": detector_host,
        "Content-Type": content_type or "application/octet-stream",
        "X-Filename": filename,
        "X-TTL": "7200",
        "X-Metadata": model,
        "X-Image-Key": image_key,
    }
    try:
        response = await httpx_client.post(
            f"{ISTIO_GATEWAY}/store_image",
            headers=headers,
            content=img,  # Sends raw binary body directly
        )
        response.raise_for_status()
        logger.info("OK image has been saved with Redis key: %s", image_key)
    except Exception:
        logger.exception("ERROR saving image with Redis key: %s", image_key)
##########################################################################################################################################
"""
curl -X POST "http://localhost:8080/predict_encoded" -H "Host: image-api.default.example.com" -F "image=@immagine.png" -F "model=simple-cnn"
"""
@app.post("/predict_encoded")
async def predict_encoded(image: UploadFile = File(...), model: str = Form(...)):
    img = await image.read() #image container the raw binary data
    image_key = str(uuid.uuid4()) #create the unique key VERY IMPORTANT FOR THE LINK OF THE SYSTEM 
    result = await forward(img, model, image.content_type or "image/png", image_key)
    #not wait for the redis response, a corutine task manage it and the client receive the inference response ass soon as possible
    asyncio.create_task(store_image_to_detector(img, image.filename, image.content_type or "image/png", model, image_key))
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"predicted_class": result["predicted_class"], "image_key": image_key}

"""
curl -X POST \                                                                               "http://localhost:8080/predict_batch_encoded" \
-H "Host: image-api.default.example.com" \
-F "files=@immagine.png" \
-F "files=@immagine.png" \
-F "models=simple-cnn,simple-cnn-test"

{"predictions":[{"model":"simple-cnn","predicted_class":2,"image_key":"800187de-892b-4c96-9cc0-90d141612499"},
{"model":"simple-cnn-test","predicted_class":2,"image_key":"bc574c41-c513-44b7-bb8e-eabdda7713a3"}]}(env) 
"""

@app.post("/predict_batch_encoded")
async def predict_batch_encoded(files: List[UploadFile] = File(...), models: str = Form(...)):
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(files) != len(model_list):
        raise HTTPException(status_code=400, detail="Images/models mismatch")
    images = await asyncio.gather(*(f.read() for f in files))
    image_keys = [str(uuid.uuid4()) for _ in range(len(files))]
    predictions = await asyncio.gather(*(
        forward(img, model, file.content_type or "image/png", key)
        for img, file, model, key in zip(images, files, model_list, image_keys)
    ))
    for img, file, model, key in zip(images, files, model_list, image_keys):
        asyncio.create_task(
            store_image_to_detector(img, file.filename, file.content_type or "image/png", model, key)
        )
    return {"predictions": predictions}

##########################################################################################################################################
"""
curl -X POST \
"http://localhost:8080/predict?model=simple-cnn" \
-H "Host: image-api.default.example.com" \
-H "Content-Type: application/octet-stream" \
--data-binary @image.raw
"""
@app.post("/predict")
async def predict(request: Request, model: str = "simple-cnn"):
    img = await request.body()
    if not img:
        raise HTTPException(status_code=400, detail="Empty body")
    image_key = str(uuid.uuid4())
    result = await forward(img, model, "application/octet-stream", image_key)
    asyncio.create_task(store_image_to_detector(img, "image.raw", "application/octet-stream", model, image_key))
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"predicted_class": result.get("predicted_class"), "image_key": image_key}
"""
curl -X POST \
"http://localhost:8080/predict_batch?models=simple-cnn,simple-cnn-test" \
-H "Host: image-api.default.example.com" \
-H "Content-Type: application/octet-stream" \
--data-binary @payload.bin
"""
@app.post("/predict_batch")
async def predict_batch(request: Request, models: str):
    body = await request.body()
    if len(body) % IMAGE_SIZE != 0:
        raise HTTPException(status_code=400, detail="Invalid payload size")
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    images = [body[i:i + IMAGE_SIZE] for i in range(0, len(body), IMAGE_SIZE)]
    if len(images) != len(model_list):
        raise HTTPException(status_code=400, detail="Images/models mismatch")
    image_keys = [str(uuid.uuid4()) for _ in range(len(images))]
    predictions = await asyncio.gather(*(
        forward(img, model, "application/octet-stream", key) 
        for img, model, key in zip(images, model_list, image_keys)
    ))
    for img, model, key in zip(images, model_list, image_keys): 
        asyncio.create_task(store_image_to_detector(img, "image.bin", "application/octet-stream", model, key))
    return {"predictions": predictions}

##########################################################################################################################################

@app.get("/health")
async def health():
    return {"status": "healthy"}