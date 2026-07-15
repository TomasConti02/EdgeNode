import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RestAPI")

ISTIO_GATEWAY = os.getenv( "ISTIO_GATEWAY", "http://istio-ingressgateway.istio-system.svc.cluster.local", )
IMAGE_SIZE = 224 * 224 * 3
httpx_client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global httpx_client
    httpx_client = httpx.AsyncClient(timeout=30.0)
    logger.info("HTTP client ready")
    yield # as every other process this code receive the OS SIGNAL of shutdown and the connection close operation will be executed before this happen 
    await httpx_client.aclose()

app = FastAPI(lifespan=lifespan)

async def forward(img: bytes, model: str, content_type: str): #the single internal http inference post request operation
    try:
        response = await httpx_client.post( #asynch http post to get the inference prediction passing trough 
            f"{ISTIO_GATEWAY}/v1/models/{model}:predict",
            headers={ "Host": f"{model}.default.example.com", "Content-Type": content_type, }, #header used by istio for internal packet forwarding
            content=img,)

        response.raise_for_status()
        logger.info("%s -> %d", model, response.status_code)
        result = response.json()

        return { "model": model, "predicted_class": result["predicted_class"], }

    except httpx.HTTPStatusError as exc:
        logger.error("%s -> %d", model, exc.response.status_code)
        return {  "model": model,  "error": exc.response.text, }

    except Exception as exc:
        logger.exception("%s failed", model)
        return {  "model": model, "error": str(exc),  }
###############################################################################################################################
"""
curl -X POST \
"http://localhost:8080/predict?model=simple-cnn" \
-H "Host: image-api.default.example.com" \
-H "Content-Type: application/octet-stream" \
--data-binary @image.raw
"""
@app.post("/predict")
async def predict(request: Request, model: str = "simple-cnn"):

    img = await request.body() #receive the http rest request for the client. THREAD does not use cpu waiting for the request

    result = await forward(  img,  model,"application/octet-stream", ) # do not block thread, manage the response as soon as the system have manage it 

    return {"predicted_class": result.get("predicted_class")}
"""
curl -X POST \
"http://localhost:8080/predict_batch_encoded" \
-H "Host: image-api.default.example.com" \
-F "files=@image.jpg" \
-F "files=@image.jpg" \
-F "models=simple-cnn,simple-cnn-test"
"""
@app.post("/predict_batch")
async def predict_batch(request: Request, models: str):

    body = await request.body()

    if len(body) % IMAGE_SIZE: #check
        raise HTTPException(400, "Invalid payload size")

    model_list =  [m.strip() for m in models.split(",") if m.strip() ]
    images = [  body[ i:i + IMAGE_SIZE ] for i in range(0, len(body), IMAGE_SIZE) ] #collect each images bytes

    if len(images) != len(model_list):
        raise HTTPException(400, "Images/models mismatch")
    
    #each forward calls creates a coroutine object. * operator unpacks the generated coroutine objects as separate arguments. asyncio.gather schedules their concurrent/parallel execution and collects the results
    predictions = await asyncio.gather(*( forward(img, model, "application/octet-stream") for img, model in zip(images, model_list) ) )

    return {"predictions": predictions}
##############################################################################################################################
"""
curl -X POST \
"http://localhost:8080/predict_encoded" \
-H "Host: image-api.default.example.com" \
-F "image=@image.jpg" \
-F "model=simple-cnn"
"""
@app.post("/predict_encoded")
async def predict_encoded( image: UploadFile = File(...), model: str = Form(...)):

    img = await image.read() #receive the http rest request for the client. THREAD does not use cpu waiting for the request

    result = await forward( img, model, image.content_type or "image/jpeg",) # do not block thread, manage the response as soon as the system have manage it 

    return { "predicted_class": result.get("predicted_class") }
"""
curl -X POST \
"http://localhost:8080/predict_batch?models=simple-cnn,simple-cnn-test" \
-H "Host: image-api.default.example.com" \
-H "Content-Type: application/octet-stream" \
--data-binary @payload.bin
"""
@app.post("/predict_batch_encoded")
async def predict_batch_encoded( files: List[UploadFile] = File(...), models: str = Form(...) ):

    model_list = [m.strip() for m in models.split(",") if m.strip()]

    if len(files) != len(model_list): #check
        raise HTTPException(400, "Images/models mismatch")
    
    images = await asyncio.gather(*(f.read() for f in files)) # file reading I-O operation have to be no blocking

    #each forward calls creates a coroutine object. * operator unpacks the generated coroutine objects as separate arguments. asyncio.gather schedules their concurrent/parallel execution and collects the results
    predictions = await asyncio.gather( *( forward( img, model, file.content_type or "image/jpeg", ) for img, file, model in zip(images, files, model_list) ))

    return {"predictions": predictions}

#################################################################################
@app.get("/health")
async def health():
    return {"status": "healthy"}
