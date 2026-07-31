import argparse
import logging
import threading
import asyncio
import uuid
import aiohttp
import httpx
import tensorflow as tf
from typing import Any, Dict, Optional
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTransformer(Model): #kserve sdk component
    def __init__(self, name, namespace,
                 broker="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker",
                 broker_host="kafka-broker-ingress.knative-eventing.svc.cluster.local",
                 ce_type="org.kubeflow.serving.inference.request",
                 istio_gateway="http://istio-ingressgateway.istio-system.svc.cluster.local"):

        super().__init__(name)
        self.ready = True
        self.count = 0
        self.name = name
        self.broker = broker
        self.broker_host = broker_host
        self.ce_type = ce_type
        self.istio_gateway = istio_gateway
        self.predictor_host = f"{name}-predictor.{namespace}.svc.cluster.local"
        
        # create an async event loop for knative eventing operations 
        self.loop = asyncio.new_event_loop()
        self.event_queue = None
        
        threading.Thread(target=self._run_worker, daemon=True).start()
        logger.info(f"Transformer [{name}] predictor={self.predictor_host}")

    def _run_worker(self):
        asyncio.set_event_loop(self.loop)
        self.event_queue = asyncio.Queue(maxsize=5000)
        self.loop.run_until_complete(self.event_worker())

    async def event_worker(self):
        connector = aiohttp.TCPConnector(limit=200)
        async with aiohttp.ClientSession(connector=connector) as session:
            while True:
                event_data = await self.event_queue.get()
                asyncio.create_task(self._send_batch(session, event_data))

    async def _send_batch(self, session, event_data):
        preds = event_data["predictions"] # inference batch possibility
        image_key = event_data["image_key"]
        
        payload = {
            "image_key": image_key,
            "instances": [p["embedding"][:4] for p in preds]  # reduce to 4 because better for debugging
        }
        
        headers = { # metadata through the header 
            "Host": self.broker_host, # used by istio for routing
            "Ce-Id": str(uuid.uuid4()), # unique id for the message event in kafka
            "Ce-Specversion": "1.0",
            "Ce-Type": self.ce_type,  # event domain -> org.kubeflow.serving.inference.request
            "Ce-Source": self.name, 
            "Content-Type": "application/json",
            "X-Image-Key": image_key  # image redis key
        }
        try:
            async with session.post(self.broker, json=payload, headers=headers, timeout=3) as response:
                response.raise_for_status()
                logger.info(f"Kafka event sent for Key: {image_key} | batch={len(preds)}")
        except Exception as e:
            logger.error(f"Kafka ERROR for key: {image_key}: {e}")
        finally:
            self.event_queue.task_done()

    async def _store_image_to_detector(self, img: bytes, filename: str, content_type: str, image_key: str):
        """Fire-and-forget: sends the raw image asynchronously to the OOD Redis detector service via Istio."""
        detector_host = f"ood-detector-{self.name}.default.example.com"
        headers = {
            "Host": detector_host,
            "Content-Type": content_type or "application/octet-stream",
            "X-Filename": filename,
            "X-TTL": "600",
            "X-Metadata": self.name,
            "X-Image-Key": image_key,
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(f"{self.istio_gateway}/store_image", headers=headers, content=img)
                response.raise_for_status()
                logger.info("OK image has been saved with Redis key: %s", image_key)
        except Exception:
            logger.exception("ERROR saving image with Redis key: %s", image_key)

    def preprocess(self, payload: bytes, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        logger.info("RAW IMAGE RECEIVED bytes=%d", len(payload))
        
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        image_key = headers.get("x-image-key") or str(uuid.uuid4())
        filename = headers.get("x-filename", "unknown.jpg")
        content_type = headers.get("content-type", "application/octet-stream")
        
        logger.info(f"Preprocess received X-Image-Key: {image_key}")

        # Fire-and-forget handoff: Dispatch image storage to the background loop without waiting
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._store_image_to_detector(payload, filename, content_type, image_key), 
                self.loop
            )

        if content_type == "application/octet-stream": # raw images 
            img_tensor = tf.io.decode_raw(payload, tf.uint8) # bypass decompression and create input 
            img_tensor = tf.reshape(img_tensor, [224, 224, 3])
        else: # png image
            img_tensor = tf.io.decode_image(payload, channels=3, expand_animations=False)
            img_tensor = tf.image.resize(img_tensor, [224, 224])

        img_tensor = tf.expand_dims(img_tensor, axis=0) # add batch dim -> [1, 224, 224, 3]
        img_tensor = tf.cast(img_tensor, tf.float32) / 255.0 

        logger.info("Model input shape=%s dtype=%s", img_tensor.shape, img_tensor.dtype)
        return { "instances": img_tensor.numpy().tolist(), "image_key": image_key }

    def postprocess(self, outputs: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        logger.info(f"Predictor output: {outputs}")
        
        headers = {k.lower(): v for k, v in (headers or {}).items()} if headers else {}
        image_key = outputs.get("image_key") or headers.get("x-image-key") or "unknown-key"
        
        preds = outputs["predictions"]
    
        event_data = { "predictions": preds, "image_key": image_key }
        
        if self.event_queue:
            asyncio.run_coroutine_threadsafe(self.event_queue.put(event_data), self.loop)
        
        pred = preds[0]
        logger.info(f"POST PROCESS {self.count} per Chiave: {image_key}")
        self.count += 1
        
        return { "predicted_class": int(pred["predicted_class"]) }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument("--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local")
    parser.add_argument("--ce_type", default="org.kubeflow.serving.inference.request")
    parser.add_argument("--istio_gateway", default="http://istio-ingressgateway.istio-system.svc.cluster.local")
    args, _ = parser.parse_known_args()

    models = [m.strip() for m in args.model_names.split(",") if m.strip()]
    if not models:
        raise ValueError("ERROR: no model names provided")
    
    transformers = [
        ImageTransformer(
            name=m, 
            namespace=args.namespace, 
            broker=args.broker, 
            broker_host=args.broker_host, 
            ce_type=args.ce_type,
            istio_gateway=args.istio_gateway
        ) for m in models
    ]

    logger.info(f"Starting transformers={len(transformers)}") 
    ModelServer().start(transformers)
