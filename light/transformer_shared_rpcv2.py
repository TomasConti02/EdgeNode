import argparse
import base64
import logging
import os
import sys
import asyncio
import uuid
import aiohttp
import cv2
import numpy as np
import grpc
from typing import Any, Dict, Optional, Union
from fastapi import Request, Response, HTTPException
from kserve import Model, ModelServer

# Import Protobuf nativi di TensorFlow Serving
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

NORM_FACTOR = np.float32(1.0 / 255.0)


class ImageTransformer(Model):
    def __init__(self, name: str, namespace: str, predictor_grpc_port: str = "9000",
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
        
        # Target gRPC nativo su localhost (TensorFlow Serving)
        self.predictor_target = f"127.0.0.1:{predictor_grpc_port}"
        
        # Inizializzati come None per garantire Lazy Initialization (Fork Safety)
        self._grpc_channel: Optional[grpc.aio.Channel] = None
        self._grpc_stub: Optional[prediction_service_pb2_grpc.PredictionServiceStub] = None
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_grpc_stub(self) -> prediction_service_pb2_grpc.PredictionServiceStub:
        """Crea il canale gRPC in maniera Lazy all'interno del processo worker corrente."""
        if self._grpc_channel is None:
            options = [
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.max_send_message_length', 100 * 1024 * 1024)
            ]
            self._grpc_channel = grpc.aio.insecure_channel(self.predictor_target, options=options)
            self._grpc_stub = prediction_service_pb2_grpc.PredictionServiceStub(self._grpc_channel)
        return self._grpc_stub

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=1000, limit_per_host=200, keepalive_timeout=120)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _send_to_kafka_safe(self, embeddings_list: list, image_key: str):
        try:
            session = self._get_session()
            if not embeddings_list:
                return
            payload = {"image_key": image_key, "instances": embeddings_list} 
            headers = { 
                "Host": self.broker_host, 
                "Ce-Id": uuid.uuid4().hex, 
                "Ce-Specversion": "1.0", 
                "Ce-Type": self.ce_type, 
                "Ce-Source": self.name, 
                "Content-Type": "application/json", 
                "X-Image-Key": image_key 
            }
            async with session.post(self.broker, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=2)):
                pass
        except Exception as e:
            logger.error("Kafka Async ERROR key %s: %s", image_key, e)

    async def _store_image_to_detector_safe(self, img: bytes, filename: str, content_type: str, image_key: str):
        try:
            session = self._get_session()
            headers = { 
                "Host": self.detector_host, 
                "Content-Type": content_type, 
                "X-Filename": filename, 
                "X-TTL": "600", 
                "X-Metadata": self.name, 
                "X-Image-Key": image_key 
            }
            url = f"{self.istio_gateway}/store_image"
            async with session.post(url, headers=headers, data=img, timeout=aiohttp.ClientTimeout(total=5)):
                pass
        except Exception as e:
            logger.error("Detector Async ERROR key %s: %s", image_key, e)

    def preprocess(self, payload: Union[bytes, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        image_key = None
        filename = "unknown.png"
        content_type = "image/png"

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
            image_key = uuid.uuid4().hex

        img_bytes = None

        # 1. Caso HTTP Raw Bytes (application/octet-stream da k6 / endpoint predict_raw)
        if isinstance(payload, (bytes, bytearray)):
            img_bytes = bytes(payload)

        # 2. Caso Payload KServe Protobuf o oggetto V2
        elif hasattr(payload, "raw_input_contents") and payload.raw_input_contents:
            raw_item = payload.raw_input_contents[0]
            img_bytes = base64.b64decode(raw_item) if isinstance(raw_item, str) else bytes(raw_item)

        # 3. Caso JSON Dictionary (POST HTTP V1/V2)
        elif isinstance(payload, dict):
            if "raw_input_contents" in payload and payload["raw_input_contents"]:
                raw_item = payload["raw_input_contents"][0]
                img_bytes = base64.b64decode(raw_item) if isinstance(raw_item, str) else bytes(raw_item)
            elif "instances" in payload and payload["instances"]:
                raw_item = payload["instances"][0]
                if isinstance(raw_item, dict) and "b64" in raw_item:
                    img_bytes = base64.b64decode(raw_item["b64"])
                elif isinstance(raw_item, str):
                    img_bytes = base64.b64decode(raw_item)

        if img_bytes is None:
            raise ValueError(f"Impossibile estrarre il buffer dell'immagine da payload di tipo: {type(payload)}")

        # Invio asincrono in background al detector
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._store_image_to_detector_safe(img_bytes, filename, content_type, image_key))
        except RuntimeError:
            pass

        # Decodifica PNG ed elaborazione OpenCV
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Errore nella decodifica dell'immagine PNG/JPEG.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        if h != 224 or w != 224:
            img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Matrice Float32 normalizzata (1, 224, 224, 3)
        img_float = np.empty((1, 224, 224, 3), dtype=np.float32)
        np.multiply(img_rgb, NORM_FACTOR, out=img_float[0])

        return {
            "tensor_data": img_float,
            "image_key": image_key
        }

    async def predict(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Invia la richiesta gRPC nativa a TensorFlow Serving su localhost:9000"""
        img_float = payload["tensor_data"]
        image_key = payload["image_key"]

        # Costruzione della PredictRequest di TF Serving
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name

        # Popolamento del TensorProto
        tensor_proto = tensor_pb2.TensorProto()
        tensor_proto.dtype = types_pb2.DT_FLOAT
        tensor_proto.tensor_shape.dim.add(size=1)
        tensor_proto.tensor_shape.dim.add(size=224)
        tensor_proto.tensor_shape.dim.add(size=224)
        tensor_proto.tensor_shape.dim.add(size=3)
        tensor_proto.tensor_content = img_float.tobytes()

        request.inputs["input"].CopyFrom(tensor_proto)

        # Chiamata gRPC asincrona
        stub = self._get_grpc_stub()
        response = await stub.Predict(request, timeout=10.0)

        # Estrazione risultati
        predicted_class = -1
        embeddings = []

        if "predicted_class" in response.outputs:
            predicted_class = int(response.outputs["predicted_class"].int_val[0])
        elif "probabilities" in response.outputs:
            probs = np.frombuffer(response.outputs["probabilities"].tensor_content, dtype=np.float32)
            predicted_class = int(np.argmax(probs))

        if "embedding" in response.outputs:
            emb_raw = np.frombuffer(response.outputs["embedding"].tensor_content, dtype=np.float32)
            embeddings = [emb_raw[:4].tolist()]

        return {
            "predicted_class": predicted_class,
            "embeddings": embeddings,
            "image_key": image_key
        }

    def postprocess(self, outputs: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        image_key = outputs.get("image_key", "unknown")
        embeddings = outputs.get("embeddings", [])
        predicted_class = outputs.get("predicted_class", -1)

        if embeddings:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._send_to_kafka_safe(embeddings, image_key))
            except RuntimeError:
                pass

        return {"predicted_class": predicted_class, "image_key": image_key}


# Custom Endpoint per accettare stream di byte PNG raw bypassando il parser JSON KServe
async def custom_predict_raw(request: Request):
    model_name = request.path_params.get("model_name")
    server = request.app.state.model_server
    model = server.registered_models.get(model_name)
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
    raw_body = await request.body()
    headers = dict(request.headers)
    
    # Esegue la pipeline completa del Transformer: preprocess -> predict -> postprocess
    preprocessed = model.preprocess(raw_body, headers=headers)
    predicted = await model.predict(preprocessed, headers=headers)
    response_data = model.postprocess(predicted, headers=headers)
    
    return response_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--predictor_grpc_port", default="9000")
    parser.add_argument("--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument("--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local")
    parser.add_argument("--ce_type", default="org.kubeflow.serving.inference.request")
    parser.add_argument("--istio_gateway", default="http://istio-ingressgateway.istio-system.svc.cluster.local")
    args, _ = parser.parse_known_args()

    models = [m.strip() for m in args.model_names.split(",") if m.strip()]
    workers = int(os.getenv("WORKERS", "8"))
    
    transformers = [
        ImageTransformer(
            name=m,
            namespace=args.namespace,
            predictor_grpc_port=args.predictor_grpc_port,
            broker=args.broker,
            broker_host=args.broker_host,
            ce_type=args.ce_type,
            istio_gateway=args.istio_gateway
        ) for m in models
    ]
    
    # Inizializza il ModelServer e registra la rotta personalizzata prima dello start
    server = ModelServer(workers=workers)
    
    # Inserimento della Custom Route FastAPI per i byte raw
    server.app.add_api_route(
        "/v1/models/{model_name}:predict_raw",
        custom_predict_raw,
        methods=["POST"]
    )
    server.app.state.model_server = server
    
    server.start(transformers)


"""
import argparse
import base64
import logging
import os
import sys
import asyncio
import uuid
import aiohttp
import cv2
import numpy as np
import grpc
from typing import Any, Dict, Optional, Union
from kserve import Model, ModelServer

# Import Protobuf nativi di TensorFlow Serving
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

NORM_FACTOR = np.float32(1.0 / 255.0)


class ImageTransformer(Model):
    def __init__(self, name: str, namespace: str, predictor_grpc_port: str = "9000",
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
        
        # Target gRPC nativo su localhost (TensorFlow Serving)
        self.predictor_target = f"127.0.0.1:{predictor_grpc_port}"
        
        # Inizializzati come None per garantire Lazy Initialization (Fork Safety)
        self._grpc_channel: Optional[grpc.aio.Channel] = None
        self._grpc_stub: Optional[prediction_service_pb2_grpc.PredictionServiceStub] = None
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_grpc_stub(self) -> prediction_service_pb2_grpc.PredictionServiceStub:
        
        if self._grpc_channel is None:
            options = [
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.max_send_message_length', 100 * 1024 * 1024)
            ]
            self._grpc_channel = grpc.aio.insecure_channel(self.predictor_target, options=options)
            self._grpc_stub = prediction_service_pb2_grpc.PredictionServiceStub(self._grpc_channel)
        return self._grpc_stub

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=1000, limit_per_host=200, keepalive_timeout=120)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _send_to_kafka_safe(self, embeddings_list: list, image_key: str):
        try:
            session = self._get_session()
            if not embeddings_list:
                return
            payload = {"image_key": image_key, "instances": embeddings_list} 
            headers = { 
                "Host": self.broker_host, 
                "Ce-Id": uuid.uuid4().hex, 
                "Ce-Specversion": "1.0", 
                "Ce-Type": self.ce_type, 
                "Ce-Source": self.name, 
                "Content-Type": "application/json", 
                "X-Image-Key": image_key 
            }
            async with session.post(self.broker, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=2)):
                pass
        except Exception as e:
            logger.error("Kafka Async ERROR key %s: %s", image_key, e)

    async def _store_image_to_detector_safe(self, img: bytes, filename: str, content_type: str, image_key: str):
        try:
            session = self._get_session()
            headers = { 
                "Host": self.detector_host, 
                "Content-Type": content_type, 
                "X-Filename": filename, 
                "X-TTL": "600", 
                "X-Metadata": self.name, 
                "X-Image-Key": image_key 
            }
            url = f"{self.istio_gateway}/store_image"
            async with session.post(url, headers=headers, data=img, timeout=aiohttp.ClientTimeout(total=5)):
                pass
        except Exception as e:
            logger.error("Detector Async ERROR key %s: %s", image_key, e)

    def preprocess(self, payload: Union[bytes, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        image_key = None
        filename = "unknown.png"
        content_type = "image/png"

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
            image_key = uuid.uuid4().hex

        # Estrazione buffer primario
        if isinstance(payload, (bytes, bytearray)):
            img_bytes = bytes(payload)
        elif hasattr(payload, "raw_input_contents") and payload.raw_input_contents:
            img_bytes = payload.raw_input_contents[0]
        elif isinstance(payload, dict) and "raw_input_contents" in payload and payload["raw_input_contents"]:
            img_bytes = payload["raw_input_contents"][0]
        else:
            raise ValueError(f"Payload non valido: {type(payload)}")

        # Se arriva come stringa Base64 (da HTTP JSON), esegui il decode in bytes
        if isinstance(img_bytes, str):
            img_bytes = base64.b64decode(img_bytes)

        # Invio asincrono in background al detector
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._store_image_to_detector_safe(img_bytes, filename, content_type, image_key))
        except RuntimeError:
            pass

        # Decodifica PNG ed elaborazione OpenCV
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Errore nella decodifica dell'immagine PNG.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        if h != 224 or w != 224:
            img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Matrice Float32 normalizzata (1, 224, 224, 3)
        img_float = np.empty((1, 224, 224, 3), dtype=np.float32)
        np.multiply(img_rgb, NORM_FACTOR, out=img_float[0])

        return {
            "tensor_data": img_float,
            "image_key": image_key
        }

    async def predict(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        
        img_float = payload["tensor_data"]
        image_key = payload["image_key"]

        # Costruzione della PredictRequest di TF Serving
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name

        # Popolamento del TensorProto
        tensor_proto = tensor_pb2.TensorProto()
        tensor_proto.dtype = types_pb2.DT_FLOAT
        tensor_proto.tensor_shape.dim.add(size=1)
        tensor_proto.tensor_shape.dim.add(size=224)
        tensor_proto.tensor_shape.dim.add(size=224)
        tensor_proto.tensor_shape.dim.add(size=3)
        tensor_proto.tensor_content = img_float.tobytes()

        request.inputs["input"].CopyFrom(tensor_proto)

        # Chiamata gRPC asincrona
        stub = self._get_grpc_stub()
        response = await stub.Predict(request, timeout=10.0)

        # Estrazione risultati
        predicted_class = -1
        embeddings = []

        if "predicted_class" in response.outputs:
            predicted_class = int(response.outputs["predicted_class"].int_val[0])
        elif "probabilities" in response.outputs:
            probs = np.frombuffer(response.outputs["probabilities"].tensor_content, dtype=np.float32)
            predicted_class = int(np.argmax(probs))

        if "embedding" in response.outputs:
            emb_raw = np.frombuffer(response.outputs["embedding"].tensor_content, dtype=np.float32)
            embeddings = [emb_raw[:4].tolist()]

        return {
            "predicted_class": predicted_class,
            "embeddings": embeddings,
            "image_key": image_key
        }

    def postprocess(self, outputs: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        image_key = outputs.get("image_key", "unknown")
        embeddings = outputs.get("embeddings", [])
        predicted_class = outputs.get("predicted_class", -1)

        if embeddings:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._send_to_kafka_safe(embeddings, image_key))
            except RuntimeError:
                pass

        return {"predicted_class": predicted_class, "image_key": image_key}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--predictor_grpc_port", default="9000")
    parser.add_argument("--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument("--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local")
    parser.add_argument("--ce_type", default="org.kubeflow.serving.inference.request")
    parser.add_argument("--istio_gateway", default="http://istio-ingressgateway.istio-system.svc.cluster.local")
    args, _ = parser.parse_known_args()

    models = [m.strip() for m in args.model_names.split(",") if m.strip()]
    workers = int(os.getenv("WORKERS", "8"))
    
    transformers = [
        ImageTransformer(
            name=m,
            namespace=args.namespace,
            predictor_grpc_port=args.predictor_grpc_port,
            broker=args.broker,
            broker_host=args.broker_host,
            ce_type=args.ce_type,
            istio_gateway=args.istio_gateway
        ) for m in models
    ]
    ModelServer(workers=workers).start(transformers)
"""