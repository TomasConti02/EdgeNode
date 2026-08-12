import argparse
import asyncio
import base64
import logging
import os
import uuid
from typing import Dict, Optional, Any

import aiohttp
import cv2
import grpc
import numpy as np

from tensorflow.core.framework import tensor_pb2

from kserve import Model, ModelServer, InferRequest, InferResponse, InferInput, InferOutput
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

NORM_FACTOR = np.float32(1.0 / 255.0)
IMAGE_SHAPE = (224, 224, 3)

DT_FLOAT = 1
DT_INT32 = 3

def numpy_to_tensor_proto(array: np.ndarray) -> tensor_pb2.TensorProto:
    array = np.asarray(array)
    tensor = tensor_pb2.TensorProto()
    
    for dim in array.shape:
        tensor.tensor_shape.dim.add(size=int(dim))

    if array.dtype == np.float32:
        tensor.dtype = DT_FLOAT
        tensor.float_val.extend(array.flatten().tolist())
    elif array.dtype == np.int32:
        tensor.dtype = DT_INT32
        tensor.int_val.extend(array.flatten().tolist())
    else:
        raise ValueError(f"Unsupported NumPy dtype: {array.dtype}")
    
    return tensor


def tensor_proto_to_numpy(tensor: tensor_pb2.TensorProto) -> np.ndarray:
    shape = [int(dim.size) for dim in tensor.tensor_shape.dim]
    
    if tensor.dtype == DT_FLOAT:
        buf = tensor.tensor_content
        array = np.frombuffer(buf, dtype=np.float32) if buf else np.asarray(tensor.float_val, dtype=np.float32)
    elif tensor.dtype == DT_INT32:
        buf = tensor.tensor_content
        array = np.frombuffer(buf, dtype=np.int32) if buf else np.asarray(tensor.int_val, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported TensorProto dtype: {tensor.dtype}")
        
    return array.reshape(shape)

class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str, broker: str, broker_host: str, ce_type: str, istio_gateway: str):
        super().__init__(name)
        self.name = name
        self.predictor_host = predictor_host
        self.ready = True

        self.broker = broker
        self.broker_host = broker_host
        self.ce_type = ce_type
        self.istio_gateway = istio_gateway
        self.detector_host = f"ood-detector-{name}.default.example.com"

        self._session: Optional[aiohttp.ClientSession] = None
        self._channel = None
        self._stub = None

        logger.error("Configured transformer '%s' for TF Serving at %s", self.name, self.predictor_host)

    def _get_tf_stub(self):
        if self._stub is None:
            self._channel = grpc.insecure_channel(self.predictor_host)
            self._stub = prediction_service_pb2_grpc.PredictionServiceStub(self._channel)
        return self._stub

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=1000, limit_per_host=200, keepalive_timeout=120)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _send_to_kafka(self, embedding: list, image_key: str):
        try:
            if not embedding:
                return
            payload = {"image_key": image_key, "instances": embedding}
            headers = {
                "Host": self.broker_host, "Ce-Id": uuid.uuid4().hex,
                "Ce-Specversion": "1.0", "Ce-Type": self.ce_type,
                "Ce-Source": self.name, "Content-Type": "application/json", "X-Image-Key": image_key
            }
            async with self._get_session().post(self.broker, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status >= 300:
                    logger.error("Kafka error HTTP %s [%s]", resp.status, image_key)
        except Exception as exc:
            logger.error("Kafka exception [%s]: %s", image_key, exc)

    async def _store_image(self, image: bytes, filename: str, content_type: str, image_key: str):
        try:
            headers = {
                "Host": self.detector_host, "Content-Type": content_type,
                "X-Filename": filename, "X-TTL": "600", "X-Metadata": self.name, "X-Image-Key": image_key
            }
            url = f"{self.istio_gateway}/store_image"
            async with self._get_session().post(url, headers=headers, data=image, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status >= 300:
                    logger.error("Detector error HTTP %s [%s]", resp.status, image_key)
        except Exception as exc:
            logger.error("Detector exception [%s]: %s", image_key, exc)

    async def preprocess(self, payload: InferRequest, headers: Optional[Dict[str, str]] = None) -> InferRequest:
        headers = headers or {}
        image_key = headers.get("x-image-key") or uuid.uuid4().hex
        filename = headers.get("x-filename", "unknown.png")
        content_type = headers.get("content-type", "image/png")

        input_tensor = payload.get_input_by_name("input")
        if not input_tensor or not input_tensor.data:
            raise ValueError("V2 input 'input' missing or empty")

        image_bytes = input_tensor.data[0]
        if isinstance(image_bytes, str):
            image_bytes = base64.b64decode(image_bytes)
        image_bytes = bytes(image_bytes)

        asyncio.create_task(self._store_image(image_bytes, filename, content_type, image_key))

        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) * NORM_FACTOR
        image = np.expand_dims(image, axis=0)

        return InferRequest(
            model_name=self.name,
            infer_inputs=[InferInput(name="input", shape=list(image.shape), datatype="FP32", data=image)],
            request_id=image_key,
        )

    async def predict(self, payload: InferRequest, headers=None, response_headers=None) -> InferResponse:
        input_tensor = payload.get_input_by_name("input")
        if input_tensor is None:
            raise ValueError("Input 'input' missing")

        data = input_tensor.data
        if data is None or (hasattr(data, "__len__") and len(data) == 0):
            raise ValueError("Input 'input' is empty")

        array = np.asarray(data, dtype=np.float32).reshape(input_tensor.shape)
        
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = "serving_default"
        request.inputs["input"].CopyFrom(numpy_to_tensor_proto(array))

        try:
            stub = self._get_tf_stub()
            tf_response = await asyncio.to_thread(stub.Predict, request, timeout=60)
        except grpc.RpcError as exc:
            logger.error("TF Serving gRPC error [%s]: code=%s details=%s", self.name, exc.code(), exc.details())
            raise
        except Exception as exc:
            logger.error("TF Serving error [%s]: %s", self.name, exc)
            raise

        outputs = []
        for name in ["probabilities", "embedding", "predicted_class"]:
            if name in tf_response.outputs:
                arr = tensor_proto_to_numpy(tf_response.outputs[name])
                if name == "predicted_class":
                    arr = arr.astype(np.int32)
                outputs.append(InferOutput(name=name, shape=list(arr.shape), datatype="INT32" if name == "predicted_class" else "FP32", data=arr))

        if not outputs:
            raise RuntimeError(f"No recognized outputs from TF Serving: {list(tf_response.outputs.keys())}")

        req_id = getattr(payload, "request_id", "")
        return InferResponse(response_id=req_id, model_name=self.name, infer_outputs=outputs)

    async def postprocess(self, response: InferResponse, headers=None, response_headers=None) -> InferResponse:
        headers = headers or {}
        
        res_id = getattr(response, "response_id", None) or uuid.uuid4().hex
        image_key = headers.get("x-image-key") or res_id

        predicted = response.get_output_by_name("predicted_class")
        if not predicted:
            raise ValueError("Output 'predicted_class' not found")
        
        predicted_class = int(predicted.as_numpy().flatten()[0])

        embedding = response.get_output_by_name("embedding")
        if embedding is not None:
            asyncio.create_task(self._send_to_kafka(embedding.as_numpy().tolist(), image_key))

        # Return a valid InferResponse containing ONLY the predicted class output, satisfying KServe V2 schema validation
        return InferResponse(
            response_id=image_key,
            model_name=self.name,
            infer_outputs=[
                InferOutput(
                    name="predicted_class",
                    shape=[1],
                    datatype="INT32",
                    data=[predicted_class]
                )
            ]
        )


# =====================================================
# MAIN ENTRYPOINT
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", required=True)
    parser.add_argument("--predictor_host", required=True)
    parser.add_argument("--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument("--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local")
    parser.add_argument("--ce_type", default="org.kubeflow.serving.inference.request")
    parser.add_argument("--istio_gateway", default="http://istio-ingressgateway.istio-system.svc.cluster.local")
    args, _ = parser.parse_known_args()

    models = [m.strip() for m in args.model_names.split(",") if m.strip()]
    if not models:
        raise ValueError("No models specified")

    workers = int(os.getenv("WORKERS", "8"))
    logger.error("Starting transformer with models=%s workers=%d", models, workers)

    transformers = [
        ImageTransformer(
            name=model, predictor_host=args.predictor_host,
            broker=args.broker, broker_host=args.broker_host,
            ce_type=args.ce_type, istio_gateway=args.istio_gateway
        ) for model in models
    ]

    ModelServer(http_port=8080, grpc_port=8081, workers=workers, enable_grpc=True).start(transformers)
