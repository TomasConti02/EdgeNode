import logging
import os
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Address of the Istio Ingress Gateway used to reach the inference services
ISTIO_GATEWAY = os.getenv( "ISTIO_GATEWAY", "http://istio-ingressgateway.istio-system.svc.cluster.local",)
# Reusable asynchronous HTTP client connection, under http there is tcp that is heavy. better reuse connection and not recreate from scratch
httpx_client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global httpx_client
    httpx_client = httpx.AsyncClient(timeout=30.0) #create the tcp reusable pool in the bootstrap 
    yield #when the applciation/process receive the shutdown signal clean up all the resources
    await httpx_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(  image: UploadFile = File(...), model: str = Form("simple-cnn"),):
    logger.info( "Predict request - Model: %s, File: %s", model, image.filename,)
    # instead of wainting for the I/O the thread keep working
    #i don't want manage the preprocessing here and lets the preporcessing part deal with this
    file_content = await image.read() # Read the uploaded image as raw bytes, in asynch way
    url = f"{ISTIO_GATEWAY}/v1/models/{model}:predict" # KServe inference endpoint.
    headers = {
        "Host": f"{model}.default.example.com", # Istio virtual host routing, used internally by istio for routing
        "Content-Type": "application/octet-stream",# binary paiload,forward the image as raw binary
    }

    try: #Istio handles the routing
        response = await httpx_client.post(  url=url, headers=headers, content=file_content,) # in asynch way
        logger.info("Status: %s", response.status_code)
        logger.info("Response: %s", response.text)
        response.raise_for_status()
        result = response.json() #parse the json response and take the prediction for the client 
        return { "predicted_class": result.get("predicted_class")}
    
    except httpx.HTTPStatusError as exc:
        logger.error( "Upstream model error (%s): %s",exc.response.status_code, exc.response.text,)
        raise HTTPException(status_code=exc.response.status_code,detail=f"Model Gateway Error: {exc.response.text}",)
    except Exception as exc:
        logger.exception("Unexpected error")
        raise HTTPException( status_code=500,detail=str(exc),)
    
@app.get("/health")
async def health():
    return {"status": "healthy"}


