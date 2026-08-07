import argparse
import logging
import os
import sys
import threading
import asyncio
import uuid
import queue  # Usa la coda C-implemented ultra veloce
import aiohttp
import cv2
import numpy as np
from typing import Any, Dict, Optional
from kserve import Model, ModelServer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

logging.basicConfig(level=logging.ERROR)
root = logging.getLogger()
root.setLevel(logging.ERROR)

for name in list(logging.root.manager.loggerDict):
    logging.getLogger(name).setLevel(logging.ERROR)

for noisy in ["kserve", "kserve.trace", "uvicorn.access", "uvicorn.error"]:
    lg = logging.getLogger(noisy)
    lg.setLevel(logging.ERROR)
    lg.propagate = False

class ErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.ERROR

if not root.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(ErrorFilter())
    root.addHandler(handler)
else:
    for handler in root.handlers:
        handler.addFilter(ErrorFilter())

logger = logging.getLogger(__name__)

NORM_FACTOR = np.float32(1.0 / 255.0)

class ImageTransformer(Model):
    def __init__(self, name: str, namespace: str, predictor_port: str,
                 broker: str = "http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker",
                 broker_host: str = "kafka-broker-ingress.knative-eventing.svc.cluster.local",
                 ce_type: str = "org.kubeflow.serving.inference.request",
                 istio_gateway: str = "http://istio-ingressgateway.istio-system.svc.cluster.local"):

        super().__init__(name)
        self.ready = True
        self.name = name
        self.broker = broker
        self.broker_host = broker_host
        self.ce_type = ce_type
        self.istio_gateway = istio_gateway
        self.detector_host = f"ood-detector-{self.name}.default.example.com"
        self.predictor_host = f"127.0.0.1:{predictor_port}" # local host beacuse the predictor is into the same pod
        self.store_queue = queue.Queue(maxsize=10000) 
        self.kafka_queue = queue.Queue(maxsize=10000)
        self.loop = asyncio.new_event_loop()
        self.session: Optional[aiohttp.ClientSession] = None
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.event_worker())

    async def event_worker(self):
        connector = aiohttp.TCPConnector(limit=300, keepalive_timeout=60, enable_cleanup_closed=True)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            asyncio.create_task(self._poll_store_queue())
            asyncio.create_task(self._poll_kafka_queue())
            while True:
                await asyncio.sleep(3600)

    async def _poll_store_queue(self): #this corutine start the storage queue listening
        while True:
            try:
                # Usa run_in_executor per non bloccare l'event loop mentre la coda è vuota
                item = await self.loop.run_in_executor(None, self.store_queue.get)#in loop receive the event 
                payload, filename, content_type, image_key = item
                asyncio.create_task(self._store_image_to_detector(payload, filename, content_type, image_key)) #create a corutine task that manage aynch the event
            except Exception as e:
                logger.error("Errore poller store: %s", e)

    async def _poll_kafka_queue(self):#this corutine start the kafka queue listening
        while True:
            try:
                event_data = await self.loop.run_in_executor(None, self.kafka_queue.get) #in loop receive the event 
                asyncio.create_task(self._send_batch(event_data)) #create a corutine task that manage aynch the event
            except Exception as e:
                logger.error("Errore poller kafka: %s", e)
    #corutine asynch operation
    async def _send_batch(self, event_data: Dict[str, Any]):
        preds = event_data["predictions"]
        image_key = event_data["image_key"]
        payload = { "image_key": image_key, "instances": [p["embedding"][:4] for p in preds if "embedding" in p] } 
        headers = { "Host": self.broker_host, "Ce-Id": str(uuid.uuid4()), "Ce-Specversion": "1.0", "Ce-Type": self.ce_type, "Ce-Source": self.name, "Content-Type": "application/json", "X-Image-Key": image_key }
        try:
            async with self.session.post(self.broker, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=3)) as response:
                response.raise_for_status()
        except Exception as e:
            logger.error("Kafka ERROR for key %s: %s", image_key, e)
    #corutine asynch operation
    async def _store_image_to_detector(self, img: bytes, filename: str, content_type: str, image_key: str):
        if not self.session:
            return
        headers = { "Host": self.detector_host, "Content-Type": content_type, "X-Filename": filename, "X-TTL": "600", "X-Metadata": self.name, "X-Image-Key": image_key, }
        url = f"{self.istio_gateway}/store_image"
        try:
            async with self.session.post(url, headers=headers, data=img, timeout=aiohttp.ClientTimeout(total=15)) as response:
                response.raise_for_status()
        except Exception as e:
            logger.error("ERROR saving image with Redis key %s: %s", image_key, e)
    #manage synch and in pipe by the main thread
    def preprocess(self, payload: bytes, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        image_key = None
        filename = "unknown.jpg"
        content_type = "application/octet-stream"
        if headers:
            for k, v in headers.items():
                lk = k.lower()
                if lk == "x-image-key":
                    image_key = v
                elif lk == "x-filename":
                    filename = v
                elif lk == "content-type":
                    content_type = v
        if not image_key:
            image_key = str(uuid.uuid4())
        try: 
            self.store_queue.put_nowait((payload, filename, content_type, image_key)) #fire and forgot to the other thread
        except queue.Full:
            logger.error("Store queue full, dropping image key: %s", image_key)
        np_arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_float = img.astype(np.float32, copy=False)
        np.multiply(img_float, NORM_FACTOR, out=img_float)
        img_batch = np.expand_dims(img_float, axis=0)
        return {"instances": img_batch.tolist(), "image_key": image_key} #the only issue cloud be the to list that can be avoided by the RPC
    #manage synch and in pipe by the main thread
    def postprocess(self, outputs: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        image_key = "unknown-key"
        if headers:
            for k, v in headers.items():
                if k.lower() == "x-image-key":
                    image_key = v
                    break
        preds = outputs["predictions"]
        try:
            self.kafka_queue.put_nowait({"predictions": preds, "image_key": image_key}) #fire and forgot to the other thread
        except queue.Full:
            logger.error("Kafka queue full, dropping event for key: %s", image_key)
        return {"predicted_class": int(preds[0]["predicted_class"])}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--predictor_port", default="8082")
    parser.add_argument("--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument("--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local")
    parser.add_argument("--ce_type", default="org.kubeflow.serving.inference.request")
    parser.add_argument("--istio_gateway", default="http://istio-ingressgateway.istio-system.svc.cluster.local")
    args, _ = parser.parse_known_args()

    models = [m.strip() for m in args.model_names.split(",") if m.strip()]
    if not models:
        raise ValueError("ERROR: no model names provided")

    transformers = [ ImageTransformer( name=m, namespace=args.namespace, predictor_port=args.predictor_port, broker=args.broker, broker_host=args.broker_host, ce_type=args.ce_type, istio_gateway=args.istio_gateway ) for m in models]
    ModelServer().start(transformers)
