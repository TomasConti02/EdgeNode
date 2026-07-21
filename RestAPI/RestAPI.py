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
# Istio Gateway - service mesh, is used to route requests internally to Knative/KServe inference services. This endpoint is the k8s  DNS name
# of the Istio Ingress Gateway and is resolved by the cluster's internal DNS system.
ISTIO_GATEWAY = os.getenv("ISTIO_GATEWAY", "http://istio-ingressgateway.istio-system.svc.cluster.local")
IMAGE_SIZE = 224 * 224 * 3 #size of the raw images in bytes 
# tcp connection pooling, avoid to open a new http connection for all the req
httpx_client: httpx.AsyncClient = None  #asynch http communciation
@asynccontextmanager
async def lifespan(app: FastAPI):
    global httpx_client
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=500)
    httpx_client = httpx.AsyncClient(limits=limits, timeout=30.0) #create the tcp/http connection poooling
    logger.info("HTTP client ready")
    yield
    await httpx_client.aclose() #clean up the network resource 
app = FastAPI(lifespan=lifespan)
##########################################################################################################################################
async def forward(img: bytes, model: str, content_type: str, image_key: str): #take the images raw ( enocded or not ) and by istio send the images to the inferense service 
    try: # content type can be ---> application/octet-stream or image/png
        headers = { 
            "Host": f"{model}.default.example.com", # this host information is used by istio gateways for internal routing to the target
            "Content-Type": content_type, # content type of the starting request image payload    --> can be application/octet-stream or image.png
            "X-Image-Key": image_key  #ood redis key because i want keep the link 
        }  
        response = await httpx_client.post( # # I/O netwrok no blocking operation . launch the http rest post request to the inference service and collect the response
            f"{ISTIO_GATEWAY}/v1/models/{model}:predict", # the call is directed to the istio gataway, asking for model predict endpoint 
            headers=headers,
            content=img, #raw images data, not overhead and other pre processing ( transformer operation)
        )
        response.raise_for_status()
        result = response.json() #parse the json response
        logger.info("%s -> %d (Redis Key: %s)", model, response.status_code, image_key)
        return {"model": model, "predicted_class": result.get("predicted_class")} #to the client i want show only model and related prediction
    
    except httpx.HTTPStatusError as exc:
        logger.error("%s -> %d (Redis Key: %s)", model, exc.response.status_code, image_key)
        return {"model": model, "error": exc.response.text}
    except Exception as exc:
        logger.exception("%s failed (Redis Key: %s)", model, image_key)
        return {"model": model, "error": str(exc)}
##########################################################################################################################################
async def store_image_to_detector( img: bytes,  filename: str,  content_type: str,  model: str,  image_key: str ): #send the same image to the redis ood service 
    detector_host = f"ood-detector-{model}.default.example.com"
    # pack metadata into http header
    headers = {
        "Host": detector_host, #the image is directed to the ood and not to the inference service 
        "Content-Type": content_type or "application/octet-stream", # image data format image/png or application/octet-stream 
        "X-Filename": filename,
        "X-TTL": "600", #ttl but is the over write by the ood
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
curl -X POST \
"http://localhost:8080/predict_encoded?model=simple-cnn" \
-H "Host: image-api.default.example.com" \
-H "Content-Type: image/png" \
--data-binary @immagine.png
{"predicted_class":2}
"""
@app.post("/predict_encoded") # this end point manage the single binary endcode png images with out rest api additionl parsing
async def predict_encoded(request: Request, model: str): #less parsing and small payload
    img = await request.body() #take the http binary body 
    if not img:
        raise HTTPException(status_code=400, detail="Empty body")
    image_key = str(uuid.uuid4())
    result = await forward(  img,  model,  request.headers.get("content-type", "image/png"),  image_key ) #now the content type of the png body image is into the header of http request
    asyncio.create_task( store_image_to_detector( img, "image.png", request.headers.get("content-type", "image/png"), model, image_key ) )
    if "error" in result:
        raise HTTPException(  status_code=500,  detail=result["error"] )
    return { "predicted_class": result["predicted_class"] }
"""
curl  -X POST \
"http://localhost:8080/predict_encoded_multipart" \
-H "Host: image-api.default.example.com" \
-F "image=@immagine.png" \
-F "model=simple-cnn"
{"predicted_class":2}
"""
@app.post("/predict_encoded_multipart") #metadata and image are into the http request body 
async def predict_encoded(image: UploadFile = File(...), model: str = Form(...)): # multipart/form-data body parsing by the rest api, 
    img = await image.read() #image container the raw binary data
    image_key = str(uuid.uuid4()) #create the unique key VERY IMPORTANT FOR THE LINK OF THE SYSTEM 
    result = await forward(img, model, image.content_type or "image/png", image_key)
    #not wait for the redis response, a corutine task manage it and the client receive the inference response ass soon as possible
    asyncio.create_task(store_image_to_detector(img, image.filename, image.content_type or "image/png", model, image_key))
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"predicted_class": result["predicted_class"]}
###########################################################################################################################################
"""
curl -X POST "http://localhost:8080/predict_batch_encoded_multipart" -H "Host: image-api.default.example.com" -F "files=@immagine.png" -F "files=@immagine.png" -F "models=simple-cnn,simple-cnn-test"
{"predictions":[{"model":"simple-cnn","predicted_class":2},{"model":"simple-cnn-test","predicted_class":2}]}
"""
@app.post("/predict_batch_encoded_multipart") # body png images parsing with multipart/form-data body
async def predict_batch_encoded(files: List[UploadFile] = File(...), models: str = Form(...)):
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(files) != len(model_list):
        raise HTTPException(status_code=400, detail="Images/models mismatch")
    images = await asyncio.gather(*(f.read() for f in files))
    image_keys = [str(uuid.uuid4()) for _ in range(len(files))]
    predictions = await asyncio.gather(*( forward(img, model, file.content_type or "image/png", key) for img, file, model, key in zip(images, files, model_list, image_keys) ))
    for img, file, model, key in zip(images, files, model_list, image_keys):
        asyncio.create_task( store_image_to_detector(img, file.filename, file.content_type or "image/png", model, key) )
    return {"predictions": predictions}
"""
curl -X POST \
"http://localhost:8080/predict_batch_encoded?models=simple-cnn,simple-cnn-test" \
-H "Host: image-api.default.example.com" \
-H "Content-Type: application/octet-stream" \
-H "X-Image-Sizes: 674,674" \
--data-binary @batch.bin
{"predictions":[{"model":"simple-cnn","predicted_class":2},{"model":"simple-cnn-test","predicted_class":2}]}
"""
@app.post("/predict_batch_encoded") #batch of png images into the body, png compression is require to send each images png compression size into the request header
async def predict_batch_encoded( request: Request, models: str ):
    body = await request.body()
    size_header = request.headers.get("X-Image-Sizes") #collect the size of each images into the header
    if not size_header:
        raise HTTPException( status_code=400,detail="Missing X-Image-Sizes header")
    sizes = [ int(x.strip())for x in size_header.split(",") ]
    model_list = [ m.strip() for m in models.split(",") if m.strip()]
    if len(sizes) != len(model_list):
        raise HTTPException( status_code=400, detail="Images/models mismatch" )
    images = []
    offset = 0
    for size in sizes:
        img = body[offset:offset+size]
        if len(img) != size:
            raise HTTPException( status_code=400, detail="Invalid image size")
        images.append(img)
        offset += size
    image_keys = [ str(uuid.uuid4()) for _ in images ]
    predictions = await asyncio.gather( *( forward( img, model, "image/png", key )  for img, model, key   in zip(images, model_list, image_keys) ))
    for img, model, key in zip( images, model_list, image_keys  ):
        asyncio.create_task( store_image_to_detector(  img, "image.png",  "image/png", model, key ))
    return {  "predictions": predictions }
##########################################################################################################################################
"""
curl -X POST \
"http://localhost:8080/predict?model=simple-cnn" \
-H "Host: image-api.default.example.com" \
-H "Content-Type: application/octet-stream" \
--data-binary @image.raw
{"predicted_class":0}
"""
@app.post("/predict")
async def predict(request: Request, model: str ): #single raw binary image, no parsing and no png compression and de compression
    img = await request.body()
    if not img:
        raise HTTPException(status_code=400, detail="Empty body")
    image_key = str(uuid.uuid4())
    result = await forward(img, model, "application/octet-stream", image_key)
    asyncio.create_task(store_image_to_detector(img, "image.raw", "application/octet-stream", model, image_key))
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"predicted_class": result.get("predicted_class") }
############################################################################################################################################
"""
curl -X POST \
"http://localhost:8080/predict_batch?models=simple-cnn,simple-cnn-test" \
-H "Host: image-api.default.example.com" \
-H "Content-Type: application/octet-stream" \
--data-binary @payload.bin
{"predictions":[{"model":"simple-cnn","predicted_class":0},{"model":"simple-cnn-test","predicted_class":0}]}
"""
@app.post("/predict_batch") 
async def predict_batch(request: Request, models: str): #batch of raw images into the body of know dim/length
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
        asyncio.create_task( store_image_to_detector(img, "image.bin", "application/octet-stream", model, key) )
    return {"predictions": predictions}
##########################################################################################################################################
@app.get("/health")
async def health():
    return {"status": "healthy"}


