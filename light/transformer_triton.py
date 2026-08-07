import argparse
import logging
import os
import sys
import asyncio
import uuid
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
        self.predictor_host = f"127.0.0.1:{predictor_port}"
        
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=500, limit_per_host=100, keepalive_timeout=60, enable_cleanup_closed=True)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _send_to_kafka_safe(self, preds: list, image_key: str):
        try:
            session = await self._get_session()
            embeddings = [p["embedding"][:4] for p in preds if "embedding" in p]
            if not embeddings:
                return
                
            payload = { "image_key": image_key, "instances": embeddings } 
            headers = { 
                "Host": self.broker_host, 
                "Ce-Id": str(uuid.uuid4()), 
                "Ce-Specversion": "1.0", 
                "Ce-Type": self.ce_type, 
                "Ce-Source": self.name, 
                "Content-Type": "application/json", 
                "X-Image-Key": image_key 
            }
            async with session.post(self.broker, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=3)) as response:
                response.raise_for_status()
        except Exception as e:
            logger.error("Kafka Fire-and-Forget ERROR for key %s: %s", image_key, e)

    async def _store_image_to_detector_safe(self, img: bytes, filename: str, content_type: str, image_key: str):
        try:
            session = await self._get_session()
            headers = { 
                "Host": self.detector_host, 
                "Content-Type": content_type, 
                "X-Filename": filename, 
                "X-TTL": "600", 
                "X-Metadata": self.name, 
                "X-Image-Key": image_key 
            }
            url = f"{self.istio_gateway}/store_image"
            async with session.post(url, headers=headers, data=img, timeout=aiohttp.ClientTimeout(total=15)) as response:
                response.raise_for_status()
        except Exception as e:
            logger.error("Detector Fire-and-Forget ERROR saving image with key %s: %s", image_key, e)

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
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        loop.create_task(self._store_image_to_detector_safe(payload, filename, content_type, image_key))

        np_arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            
        img_float = img.astype(np.float32, copy=False)
        np.multiply(img_float, NORM_FACTOR, out=img_float)
        img_batch = np.expand_dims(img_float, axis=0)

        return {
            "instances": img_batch.tolist(),
            "image_key": image_key
        }

    async def predict(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        image_key = payload.get("image_key", "unknown-key")
        instances = payload.get("instances")

        triton_payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [len(instances), 224, 224, 3],
                    "datatype": "FP32",
                    "data": np.array(instances, dtype=np.float32).flatten().tolist()
                }
            ]
        }

        url = f"http://{self.predictor_host}/v2/models/{self.name}/infer"
        session = await self._get_session()
        
        async with session.post(url, json=triton_payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Triton v2 inference failed with status {response.status}: {text}")
            res_json = await response.json()
            
        outputs_map = {}
        for output in res_json.get("outputs", []):
            outputs_map[output.get("name")] = output.get("data")

        predictions = []
        pred_classes = outputs_map.get("predicted_class", [0])
        embeddings = outputs_map.get("embedding", [])
        
        batch_size = len(pred_classes)
        embed_dim = 512

        for i in range(batch_size):
            item = {"predicted_class": pred_classes[i]}
            if embeddings:
                start_idx = i * embed_dim
                end_idx = start_idx + embed_dim
                item["embedding"] = embeddings[start_idx:end_idx]
            predictions.append(item)

        return {
            "predictions": predictions,
            "image_key": image_key
        }

    def postprocess(self, outputs: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        image_key = "unknown-key"
        if headers:
            for k, v in headers.items():
                if k.lower() == "x-image-key":
                    image_key = v
                    break
        preds = outputs["predictions"]
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        loop.create_task(self._send_to_kafka_safe(preds, image_key))
            
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
        
    workers = int(os.getenv("WORKERS", "4"))
    transformers = [ ImageTransformer( name=m, namespace=args.namespace, predictor_port=args.predictor_port, broker=args.broker, broker_host=args.broker_host, ce_type=args.ce_type, istio_gateway=args.istio_gateway ) for m in models]
    ModelServer(workers=workers).start(transformers)