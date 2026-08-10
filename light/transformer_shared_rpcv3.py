import argparse
import base64
import logging
import os
import asyncio
import uuid
import aiohttp
import cv2
import numpy as np
import grpc
from typing import Any, Dict, Optional, Union
from fastapi import Request, Response, HTTPException
from kserve import Model, ModelServer
from kserve.model_server import app

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
        self.predictor_target = f"127.0.0.1:{predictor_grpc_port}" # local host and gRPC communication optimization
        
        self._grpc_channel: Optional[grpc.aio.Channel] = None
        self._grpc_stub: Optional[prediction_service_pb2_grpc.PredictionServiceStub] = None
        self._session: Optional[aiohttp.ClientSession] = None

        self.base_tensor_proto = tensor_pb2.TensorProto() #standard tensorflow gRPC binary format
        self.base_tensor_proto.dtype = types_pb2.DT_FLOAT #define the data type
        for dim_size in [1, 224, 224, 3]: #define the shape of the tensor
            self.base_tensor_proto.tensor_shape.dim.add(size=dim_size)
        #self.stub = self._get_grpc_stub() #take the predictor stub
    #create and conf  the gRPC with the predictor, gRPC is the connector between transformer and tensorflow predictor
    def _get_grpc_stub(self) -> prediction_service_pb2_grpc.PredictionServiceStub: #patter lazy, at init create the connection gRPC
        if self._grpc_channel is None:
            options = [ ('grpc.max_receive_message_length', 100 * 1024 * 1024),  ('grpc.max_send_message_length', 100 * 1024 * 1024) ] #100Mb of binary payload
            self._grpc_channel = grpc.aio.insecure_channel(self.predictor_target, options=options) #create an asynch gRPC connection 
            self._grpc_stub = prediction_service_pb2_grpc.PredictionServiceStub(self._grpc_channel) #take the communication  stub 
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
            headers = {"Host": self.broker_host,"Ce-Id": uuid.uuid4().hex,"Ce-Specversion": "1.0",
                       "Ce-Type": self.ce_type,"Ce-Source": self.name,"Content-Type": "application/json","X-Image-Key": image_key }
            async with session.post(self.broker, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=15)):
                pass
        except Exception as e:
            logger.error("Kafka Async ERROR key %s: %s", image_key, e)

    async def _store_image_to_detector_safe(self, img: bytes, filename: str, content_type: str, image_key: str):
        try:
            session = self._get_session()
            headers = { "Host": self.detector_host, "Content-Type": content_type,"X-Filename": filename,"X-TTL": "600","X-Metadata": self.name,"X-Image-Key": image_key }
            url = f"{self.istio_gateway}/store_image"
            async with session.post(url, headers=headers, data=img, timeout=aiohttp.ClientTimeout(total=15)):
                pass
        except Exception as e:
            logger.error("Detector Async ERROR key %s: %s", image_key, e)
 
    def preprocess(self, payload: Union[bytes, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        headers = headers or {}
        image_key = headers.get("x-image-key") or uuid.uuid4().hex
        filename = headers.get("x-filename", "unknown.png")
        content_type = headers.get("content-type", "image/png")
        try:
            raw_item = payload["raw_input_contents"][0]
            img_bytes = base64.b64decode(raw_item) if isinstance(raw_item, str) else bytes(raw_item)
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError("Payload ERROR") from e
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._store_image_to_detector_safe(img_bytes, filename, content_type, image_key))
        except RuntimeError:
            pass
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("ERROR image decode")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != (224, 224):
            img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        img_float = np.empty((1, 224, 224, 3), dtype=np.float32)
        np.multiply(img_rgb, NORM_FACTOR, out=img_float[0])

        return { "tensor_data": img_float, "image_key": image_key }
    
    async def predict(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        img_float = payload["tensor_data"]
        image_key = payload["image_key"]
        # create the right tensorflow Protobuf binary request format
        request = predict_pb2.PredictRequest() #first wrapper 
        request.model_spec.name = self.name #name of the inference model 
        tensor_proto = tensor_pb2.TensorProto() # second tensor wrapper 
        tensor_proto.CopyFrom(self.base_tensor_proto)
        tensor_proto.tensor_content = img_float.tobytes() #crate
        request.inputs["input"].CopyFrom(tensor_proto) #load input binary tensor for the model

        stub = self._get_grpc_stub()
        response = await stub.Predict(request, timeout=10.0) #gRPC aynch call to the predictor

        predicted_class = int(response.outputs["predicted_class"].int_val[0])
        emb_raw = response.outputs["embedding"].float_val
        embeddings = [list(emb_raw[:4])]
        #embeddings = [list(emb_raw)] all 512 emebeddings values
        return { "predicted_class": predicted_class, "embeddings": embeddings, "image_key": image_key }
    
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
    parser.add_argument("--model_names", required=True) #model name
    parser.add_argument("--namespace", default="default") #k8s deployment name space 
    parser.add_argument("--predictor_grpc_port", default="9000") #port fo gRPC
    parser.add_argument("--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument("--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local")
    parser.add_argument("--ce_type", default="org.kubeflow.serving.inference.request")
    parser.add_argument("--istio_gateway", default="http://istio-ingressgateway.istio-system.svc.cluster.local")
    args, _ = parser.parse_known_args()
    models = [m.strip() for m in args.model_names.split(",") if m.strip()]
    workers = int(os.getenv("WORKERS", "4")) #set up the parallel workers 
    transformers = [ ImageTransformer( name=m, namespace=args.namespace, predictor_grpc_port=args.predictor_grpc_port, broker=args.broker,
                                       broker_host=args.broker_host, ce_type=args.ce_type, istio_gateway=args.istio_gateway) for m in models ]
    ModelServer(workers=workers).start(transformers) #initialized the engine







"""
ERROR:__mp_main__:DEBUG -  {
  key: "probabilities"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 10
      }
    }
    float_val: 0.00319806021
    float_val: 1.07854987e-06
    float_val: 0.648779
    float_val: 0.0268888697
    float_val: 0.00986028649
    float_val: 0.000152019711
    float_val: 0.00571593409
    float_val: 0.00132279412
    float_val: 0.133430809
    float_val: 0.170651153
  }
}
outputs {
  key: "predicted_class"
  value {
    dtype: DT_INT32
    tensor_shape {
      dim {
        size: 1
      }
    }
    int_val: 2
  }
}
outputs {
  key: "embedding"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 512
      }
    }
    float_val: 0
    float_val: 0.134895
    float_val: 7.39281034
    float_val: 0.0490613244
    float_val: 0
2026-08-10 12:21:54.771 uvicorn.access INFO:     10.42.0.251:0 74 - "POST /v1/models/simple-cnn%3Apredict HTTP/1.1" 200 OK
    float_val: 0.638588607
    float_val: 1.05838692
    float_val: 5.32261801
    float_val: 0.41967684
    float    float_val: 4.67811537
    float_val: 0.0371936187
  }
}
model_spec {
  name: "simple-cnn"
  version {
    value: 1
  }
  signature_name: "serving_default"
}

2026-08-10 12:21:54.771 74 kserve.trace requestId: 1cddf90f-cd85-4a20-8ab7-f5a35b36abfe, preprocess_ms: 1.295804977, explain_ms: 0, predict_ms: 844.575643539, postprocess_ms: 0.039815903
2026-08-10 12:21:54.771 74 kserve.trace kserve.io.kserve.protocol.rest.v1_endpoints.predict: 0.8487443923950195 ['http_status:200', 'http_method:POST', 'time:wall']
2026-08-10 12:21:54.771 74 kserve.trace kserve.io.kserve.protocol.rest.v1_endpoints.predict: 0.02335300000000018 ['http_status:200', 'http_method:POST', 'time:cpu']
"""

"""
    async def predict(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        img_float = payload["tensor_data"]
        image_key = payload["image_key"]

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name

        tensor_proto = tensor_pb2.TensorProto()
        tensor_proto.CopyFrom(self.base_tensor_proto)
            
        tensor_proto.tensor_content = img_float.tobytes()
        request.inputs["input"].CopyFrom(tensor_proto)
        stub = self._get_grpc_stub()
        response = await stub.Predict(request, timeout=10.0)
        logger.error("DEBUG - Risposta gRPC grezza: %s", response)
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

        return { "predicted_class": predicted_class, "embeddings": embeddings, "image_key": image_key }
"""
"""
    def preprocess(self, payload: Union[bytes, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        image_key = headers.get("x-image-key") if headers else None
        if not image_key:
            image_key = uuid.uuid4().hex #image traker by key

        filename = headers.get("x-filename", "unknown.png") if headers else "unknown.png"
        content_type = headers.get("content-type", "image/png") if headers else "image/png"

        img_bytes = None
        if isinstance(payload, dict) and "raw_input_contents" in payload and payload["raw_input_contents"]:
            raw_item = payload["raw_input_contents"][0]
            img_bytes = base64.b64decode(raw_item) if isinstance(raw_item, str) else bytes(raw_item)
        elif hasattr(payload, "raw_input_contents") and payload.raw_input_contents:
            raw_item = payload.raw_input_contents[0]
            img_bytes = base64.b64decode(raw_item) if isinstance(raw_item, str) else bytes(raw_item)

        if img_bytes is None:
            raise ValueError("Payload ERROR")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._store_image_to_detector_safe(img_bytes, filename, content_type, image_key)) #fire and forgot
        except RuntimeError:
            pass

        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Errore nella decodifica dell'immagine.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[:2] != (224, 224):
            img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        img_float = np.empty((1, 224, 224, 3), dtype=np.float32)
        np.multiply(img_rgb, NORM_FACTOR, out=img_float[0])

        return { "tensor_data": img_float, "image_key": image_key }
    """



#######################################################################################################################################################################################################à

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
from fastapi import Request, Response, HTTPException
from kserve import Model, ModelServer
from kserve.model_server import app

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