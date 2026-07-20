import argparse
import logging
import threading
import asyncio
import uuid
import aiohttp
import tensorflow as tf
from typing import Any, Dict, Optional
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTransformer(Model): #kserve sdk component
    def __init__(self, name, namespace,
                 broker="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker",
                 broker_host="kafka-broker-ingress.knative-eventing.svc.cluster.local",
                 ce_type="org.kubeflow.serving.inference.request"):

        super().__init__(name)
        self.ready = True
        self.count = 0
        self.name = name
        self.broker = broker
        self.broker_host = broker_host
        self.ce_type = ce_type
        self.predictor_host = f"{name}-predictor.{namespace}.svc.cluster.local"
        # create an asynch event loop for knative eventing operations 
        self.loop = asyncio.new_event_loop()
        self.event_queue = None
        threading.Thread(target=self._run_worker, daemon=True).start()
        logger.info(f"Transformer [{name}] predictor={self.predictor_host}")

    def _run_worker(self):
        asyncio.set_event_loop(self.loop)
        self.event_queue = asyncio.Queue()
        self.loop.run_until_complete(self.event_worker())

    async def event_worker(self):
        async with aiohttp.ClientSession() as session:
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
        
        headers = { #metadata trough the header 
            "Host": self.broker_host, #used by istio for routing
            "Ce-Id": str(uuid.uuid4()), # unique id for the message event in kafka
            "Ce-Specversion": "1.0",
            "Ce-Type": self.ce_type,  #event domain -> org.kubeflow.serving.inference.request
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

    def preprocess(self, payload: bytes, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        logger.info("RAW IMAGE RECEIVED bytes=%d", len(payload))
        
        headers = headers or {}
        image_key = headers.get("x-image-key") or headers.get("X-Image-Key") or "unknown-key"
        logger.info(f"Preprocess received X-Image-Key: {image_key}")

        content_type = headers.get("content-type", "").lower()

        if content_type == "application/octet-stream": #raw images 
            img_tensor = tf.io.decode_raw(payload, tf.uint8)
            img_tensor = tf.reshape(img_tensor, [224, 224, 3])
        else: # png image
            img_tensor = tf.io.decode_image(payload, channels=3, expand_animations=False)

        img_tensor = tf.expand_dims(img_tensor, axis=0)
        img_tensor = tf.image.rgb_to_grayscale(img_tensor)#mock
        img_tensor = tf.image.resize(img_tensor, [28, 28]) #mock
        img_tensor = tf.cast(img_tensor, tf.float32) / 255.0 #if the model input activation change i have also to change this 
 
        return { "instances": img_tensor.numpy().tolist(), "image_key": image_key  }
    """
    there is a logic error hear -> this component have the possibility to manage inference emd batch bu use ONLY ONE REDIS KEY !!!
    for this prototype it is fine because the output emd embedding keep only one image !!!!! 
    """
    def postprocess(self, outputs: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        logger.info(f"Predictor output: {outputs}")
        
        image_key = outputs.get("image_key") or (headers.get("x-image-key") if headers else None) or "unknown-key" #the key came from the header
        
        preds = outputs["predictions"]
    
        event_data = { "predictions": preds, "image_key": image_key  }
        #async kafka event 
        asyncio.run_coroutine_threadsafe( self.event_queue.put(event_data) , self.loop) #kafka have to create an asynch msg for the ood with model emd prediction and redis key
        
        pred = preds[0]
        logger.info(f"POST PROCESS {self.count} per Chiave: {image_key}")
        self.count += 1
        #back to the rest api 
        return {  "predicted_class": int(pred["predicted_class"]),  "probabilities": pred["probabilities"], "embedding": pred["embedding"], "image_key": image_key }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument("--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local")
    parser.add_transformersargument("--ce_type", default="org.kubeflow.serving.inference.request")
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
            ce_type=args.ce_type
        ) for m in models
    ]

    logger.info(f"Starting transformers={len(transformers)}") 
    ModelServer().start(transformers) #get in a list of possibile model-transformer
