import argparse
import base64
import logging
import threading
import asyncio
import uuid
import aiohttp
import tensorflow as tf

from kserve import Model, ModelServer
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom KServe Transformer, it execute a pre and post processing o images and execute the model predictions
# asynchronously dispatches the prediction embeddings to a Knative / Kafka Eventing Broker by CloudEvents
class ImageTransformer(Model):
    def __init__(self, name: str, namespace: str, predictor_port: int = 8080,
                 broker: str = "http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker",
                 broker_host: str = "kafka-broker-ingress.knative-eventing.svc.cluster.local",
                 ce_type: str = "org.kubeflow.serving.inference.request"):
        
        super().__init__(name)
        self.ready = True
        self.count = 0

        self.broker = broker
        self.broker_host = broker_host
        self.ce_type = ce_type

        self.predictor_port = predictor_port
        self.predictor_host = f"{name}-predictor.{namespace}.svc.cluster.local"

        # Sending data to Kafka over HTTP can introduce network latency, 
        # doing it directly inside postprocess would slow down user response times
        # The code uses a dedicated background working thread to handle Kafka side-effects
        #sets a multi-threaded asynchronous 
        self.loop = asyncio.new_event_loop() #create a event loop engine
        #remember in prodaction fix a size of queue to avoid memory issue
        self.event_queue = asyncio.Queue(loop=self.loop) #now the loop engine is attach to a FIFO queue
        self.worker_thread = threading.Thread(target=self._run_worker, daemon=True) #back ground and asynch thread that handle network
        self.worker_thread.start()

        logger.info(f"==> Model [{name}] Predictor Host: {self.predictor_host}:{self.predictor_port}")
    #################################### NETWORK THREAD #################################################################
    def _run_worker(self): #build the asynch system 
        asyncio.set_event_loop( self.loop ) # assign the event loop to the asynch working thread 
        self.loop.run_until_complete( self.event_worker() ) # start the loop and pass the main corutine loop deamon task

    async def event_worker(self): #allow to manage all the msg dropped into the queue by the corutine launch from the posprocess
        pending_tasks = set()
        # Opens a single, reusable pool of HTTP connections 
        async with aiohttp.ClientSession() as session:
            while True: #loop
                # Non-blocking pause. Thread "sleeps" here using 0% CPU until a message drops into the queue
                preds = await self.event_queue.get()
                # corutine_finger_print=self._send_batch(session, preds)  -> crea the corutine
                #task = asyncio.create_task (corutine_finger_print)-> schedule the corutine and creating a task7
                # task is schedule into the event loop for a concorrent execution. The event loop engine accepts the task, puts it into its active calendar, and executes it concurrently whenever other tasks are paused or waiting for the network
                task = asyncio.create_task( self._send_batch(session, preds) ) # do not wait for the response !!! fast fire-and-forget
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard) #clean up the back ground task state
    # the original thread loop can yield back the control as soon as there is a network task wait
    async def _send_batch(self, session: aiohttp.ClientSession, preds: List[Dict]): #asynch func, it is a corutine
        #payload = {"instances": [p["embedding"] for p in preds]}
        payload = {"instances": [p["embedding"][:4] for p in preds]} 
        headers = {
            "Host": self.broker_host,
            "Ce-Id": str(uuid.uuid4()),
            "Ce-Specversion": "1.0",
            "Ce-Type": self.ce_type,
            "Ce-Source": self.name,
            "Content-Type": "application/json",
        }
        try:
            #execute the asynch network call to kafka
            async with session.post(self.broker, json=payload, headers=headers, timeout=3) as resp:
                resp.raise_for_status()
                logger.info(f"Kafka event sent (batch of {len(preds)})")
        except Exception as e:
            logger.error(f"Kafka error: {e}")
        finally: #run in case of success or not
            self.event_queue.task_done() #best practise
    #######################################################################################################################
    def preprocess(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, List[List[float]]]:
        processed = []
        for inst in payload["instances"]:
            img = tf.image.decode_image(base64.b64decode(inst["image"]["b64"]), channels=3)
            img = tf.image.rgb_to_grayscale(img)
            img = tf.image.resize(img, [28, 28])
            img = tf.cast(img, tf.float32) / 255.0
            processed.append(img.numpy().tolist())
        return {"instances": processed}

    def postprocess(self, outputs: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        preds = outputs["predictions"]
        # self.event_queue.put(preds) is the coroutine object and the exact operation (the instruction to add data to the queue)
        # It is executed entirely by the background working thread
        #It uses self.loop as the engine to drive that execution
        # it does the same operation as task = asyncio.create_task( self._send_batch(session, preds) )  BUT into the event engine loop of the working thread 
        asyncio.run_coroutine_threadsafe( self.event_queue.put(preds), self.loop ) #bridge between two threads

        pred = preds[0]
        logger.info(f"POST PROCESS {self.count}")
        self.count += 1
        return {
            "predicted_class": int(pred["predicted_class"]),
            "probabilities": pred["probabilities"],
            "embedding": pred["embedding"],
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_names", required=True, help="List of models separated by commas -> model1,model2,model3")
    parser.add_argument("--namespace", default="default", help="K8s Namespace")
    parser.add_argument("--predictor_port", type=int, default=8080)
    parser.add_argument("--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument("--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local")
    parser.add_argument("--ce_type", default="org.kubeflow.serving.inference.request")

    args, _ = parser.parse_known_args()

    models_to_serve = [name.strip() for name in args.model_names.split(",") if name.strip()]
    if not models_to_serve:
        raise ValueError("ERROR: no model names provided")

    instances_list = []
    for model_name in models_to_serve:
        transformer = ImageTransformer( name=model_name, namespace=args.namespace, predictor_port=args.predictor_port, broker=args.broker, broker_host=args.broker_host,ce_type=args.ce_type,)
        instances_list.append(transformer)

    logger.info(f"ModelServer started, {len(instances_list)} Transformer(s) loaded")
    ModelServer().start(instances_list)