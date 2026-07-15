import argparse
import logging
import threading
import asyncio
import uuid
import aiohttp
import io
import tensorflow as tf
from PIL import Image
from typing import Any, Dict, Optional
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTransformer(Model):
    def __init__( self, name, namespace,
        broker="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker",
        broker_host="kafka-broker-ingress.knative-eventing.svc.cluster.local",
        ce_type="org.kubeflow.serving.inference.request" ):

        super().__init__(name)
        self.ready = True
        self.count = 0
        self.name = name
        self.broker = broker
        self.broker_host = broker_host
        self.ce_type = ce_type
        self.predictor_host = ( f"{name}-predictor.{namespace}.svc.cluster.local")
        # Background Kafka worker because prediction and Kafka communication execute independently
        # inference thread never executes asynchronous networking
        self.loop = asyncio.new_event_loop() #asyncio event loop
        self.event_queue = None
        threading.Thread( target=self._run_worker, daemon=True ).start() #start the asynch system into a asynch working thread
        logger.info( f"Transformer [{name}] predictor={self.predictor_host}" )
    """
    KServe handles incoming predictions req using its main server thread.
    Pushing event processing off to a dedicated event loop ensures that slow Kafka brokers or network timeouts will not block the main prediction engine
    """
    def _run_worker(self): # aynch  kafka worker initialization
        asyncio.set_event_loop(self.loop) # bind the event loop to working thread
        self.event_queue = asyncio.Queue() # creates an asynchronous queue for the event loop
        self.loop.run_until_complete( self.event_worker() ) #start the loop 

    async def event_worker(self): #asynch corutine
        async with aiohttp.ClientSession() as session: #create a reusable connection pool (tcp connection creation is a bottleneck)
            while True: #into the loop TCP connection pool remains alive and cached in memory
                preds = await self.event_queue.get() #asynch  and NOT BLOCKING for the working thread 
                asyncio.create_task( self._send_batch( session, preds ) ) # new corutine

    async def _send_batch( self, session, preds ): #asynch corutine
        payload = { "instances":[ p["embedding"][:4] for p in preds ] } # why 4 ? -> because mock, simpler read the kafka topic log
        headers = {"Host": self.broker_host,"Ce-Id": str(uuid.uuid4()), "Ce-Specversion": "1.0", "Ce-Type": self.ce_type,"Ce-Source": self.name, "Content-Type": "application/json" }
        try: #networking operation by the corutine
            async with session.post( self.broker, json=payload, headers=headers, timeout=3 ) as response:
                response.raise_for_status()
                logger.info( f"Kafka event sent batch={len(preds)}" )
        except Exception as e:
            logger.error( f"Kafka error: {e}" )
        finally:
            self.event_queue.task_done() #mark the corutine task done
    # http header request -> Content-Type: application/octet-stream | Content-Type: image/jpeg
    def preprocess( self, payload: bytes, headers: Optional[Dict[str, str]] = None, ): #mange the pre processing of the http inference request from the rest api through istio
        logger.info("RAW IMAGE RECEIVED bytes=%d", len(payload))

        content_type = ( headers.get("content-type", "") if headers else "" ).lower() # read the request content type 

        if content_type == "application/octet-stream": #  Raw RGB uint8: 224 x 224 x 3
            img_tensor = tf.io.decode_raw(payload, tf.uint8) #read the raw payload and creat a uint8 array of images byte 
            img_tensor = tf.reshape(img_tensor, [224, 224, 3]) # create the standard inference input images
        else:# JPEG / PNG / BMP / GIF
            img_tensor = tf.io.decode_image( payload, channels=3, expand_animations=False,  ) #de-compless the payload according to the image input formate

        img_tensor = tf.expand_dims(img_tensor, axis=0) # create a batch -> batch x height x width x channels = 1 x 224 x 224 x 3
        ##### the following tf operation are executed because mock inference model compatibility ################################################
        img_tensor = tf.image.rgb_to_grayscale(img_tensor)
        img_tensor = tf.image.resize(img_tensor, [28, 28])
        img_tensor = tf.cast(img_tensor, tf.float32) / 255.0

        return {    "instances": img_tensor.numpy().tolist()}
        
    def postprocess( self, outputs: Dict[str,Any], headers=None ):
        logger.info( f"Predictor output: {outputs}" )
        preds = outputs["predictions"]
        # pass the embedding prediction to the queue of the worker thread that is listening with a even loop
        asyncio.run_coroutine_threadsafe(  self.event_queue.put(preds),  self.loop )  # async kafka dispatch embedding results, Zero Latency
        pred = preds[0]
        logger.info(  f"POST PROCESS {self.count}")
        self.count += 1
        return { "predicted_class": int(pred["predicted_class"]), "probabilities":pred["probabilities"], "embedding":pred["embedding"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(  "--model_names",  required=True)
    parser.add_argument("--namespace",default="default")
    parser.add_argument(  "--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument( "--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local" )
    parser.add_argument( "--ce_type", default="org.kubeflow.serving.inference.request" )
    args,_ = parser.parse_known_args()

    models=[ m.strip() for m in args.model_names.split(",") if m.strip() ]
    if not models:
        raise ValueError("ERROR: no model names provided")
    
    transformers=[  ImageTransformer( name=m, namespace=args.namespace, broker=args.broker, broker_host=args.broker_host, ce_type=args.ce_type ) for m in models ]

    logger.info(  f"Starting transformers={len(transformers)}" )
    ModelServer().start(transformers)
