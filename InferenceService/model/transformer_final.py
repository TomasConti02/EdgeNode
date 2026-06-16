import argparse
import base64
import logging
import os
import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
"""
Kserve have a main CRD InferenceService composed by :
1. Predicotor, executor of inference model 
2. Transformer of the model that execute pre processing and post processing
"""
class ImageTransformer(Model):
    def __init__(self, name: str, namespace: str, predictor_port: int = 8080):
        super().__init__(name)
        self.ready = True

        #Transformer have to know the predictor logic endpoint 
        self.predictor_port = predictor_port
        self.predictor_host = f"{name}-predictor.{namespace}.svc.cluster.local" # dynamic inference service resolution acoording to Knative URL policy
        logger.info(f"==> Model [{name}] conf and Predictor Host: {self.predictor_host}:{self.predictor_port}")

    def preprocess(  self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None ) -> Dict[str, List[List[float]]]:
        #pre processing example of task, receive the @json image into the payload and coded into base64
        #  decode the  b64, conver in gray scale, resize int 28X28 and normalize before sending to model 
        processed = []
        for inst in payload["instances"]:
            img = tf.image.decode_image(base64.b64decode(inst["image"]["b64"]), channels=3)
            img = tf.image.rgb_to_grayscale(img)
            img = tf.image.resize(img, [28, 28])
            img = tf.cast(img, tf.float32) / 255.0
            processed.append(img.numpy().tolist())
        return {"instances": processed}

    def postprocess( self, outputs: Dict[str, Any], headers: Optional[Dict[str, str]] = None ) -> Dict[str, Any]: # in post processing parsing the inference model output 
        pred = outputs["predictions"][0]
        return { "predicted_class": int(pred["predicted_class"]), "probabilities": pred["probabilities"], "embedding": pred["embedding"], }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", required=True, help="List of models and , as a splitter -> model1, model2, model3")
    parser.add_argument("--namespace", default="default", help="K8s Namespace")
    parser.add_argument("--predictor_port", type=int, default=8080) 
    args, _ = parser.parse_known_args()
    # model list name parsing base on ,
    models_to_serve = [name.strip() for name in args.model_names.split(",") if name.strip()]
    
    if not models_to_serve:
        raise ValueError("ERROR in the models name list param ")

    # ImageTransformer for each inference CNN service 
    instances_list = []
    for model_name in models_to_serve:
        transformer_instance = ImageTransformer( name=model_name, namespace=args.namespace,  predictor_port=args.predictor_port )
        instances_list.append(transformer_instance)

    logger.info(f"ModelServer started, {len(instances_list)} Transformer loaded")
    #even if we use a list of model kserve create a trasformer pod for each one
    ModelServer().start(instances_list) # we have to pass the entire list to ModelServer

"""
http://localhost:8080 incress point becasue the port forwarding
header Host: simple-cnn.default.example.com ->used by istio gataway to route and rach the inference service, requesto to InferenceService simple-cnn in name space default

gataway send the request to a trasformer before like simple-cnn-transformer.default.svc.cluster.local

curl -X POST http://localhost:8080/v1/models/simple-cnn:predict \
     -H "Host: simple-cnn.default.example.com" \
     -H "Content-Type: application/json" \
     -d @image.json
##########################################################################
### -> /v1/models/simple-cnn:predict it is the resource URI into server
### -> simple-cnn.default.example.com the gataway that receive the post re router all to this service 
kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80
"""


"""
transformer:
    containers:
    - name: kserve-container
      image: tomasconti02/image-transformer:v14   
      command: ["python", "-m", "transformer"]
      args:
      - --model_names
      - "simple-cnn,simple-cnn-test"               
      - --namespace
      - "default"                                

"""
