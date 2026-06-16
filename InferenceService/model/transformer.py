import argparse
import base64
import logging
import os
import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer
################################ "transformer_image": "tomasconti02/image-transformer:v14"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTransformer(Model):
    def __init__(self, name: str, namespace: str, predictor_port: int = 8080):
        # Passiamo il nome del modello specifico alla classe padre Model
        super().__init__(name)
        self.ready = True
        self.predictor_port = predictor_port

        # RISOLUZIONE DINAMICA DEL DNS INTERNO:
        # Usiamo l'URL stabile di Knative (senza la revisione -00001) per massima resilienza agli aggiornamenti
        self.predictor_host = f"{name}-predictor.{namespace}.svc.cluster.local"
        
        logger.info(f"==> Modello [{name}] configurato con Predictor Host: {self.predictor_host}:{self.predictor_port}")

    def preprocess(self, payload, headers=None):
        if isinstance(payload, list):
            instances = payload
        elif isinstance(payload, dict):
            instances = payload.get("instances", [])
        else:
            raise ValueError("Unsupported payload format")
        
        if not instances:
            return {"instances": []}
        
        processed = []
        for instance in instances:
            if isinstance(instance, list):
                instance = instance[0]

            img_b64 = instance["image"]["b64"]
            img_bytes = base64.b64decode(img_b64)
            
            # Pipeline di trasformazione immagini (TensorFlow)
            image = tf.image.decode_image(img_bytes, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            image = tf.cast(image, tf.float32) / 255.0

            processed.append(image.numpy().tolist())

        return {"instances": processed}

    def postprocess(self, outputs, headers=None):
        if isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, dict):
                    outputs = item
                    break
        
        if isinstance(outputs, dict):
            data = outputs.get("predictions", outputs)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            probs = data.get("probabilities")
            embedding = data.get("embedding")
            pred_class = data.get("predicted_class")
        else:
            probs = embedding = pred_class = None
        
        # Safe-checking e serializzazione standard JSON
        if probs is not None:
            if hasattr(probs, 'tolist'):
                probs = probs.tolist()
            elif isinstance(probs, np.ndarray):
                probs = probs.tolist()
        
        if embedding is not None:
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
        
        if pred_class is not None:
            pred_class = int(pred_class)
        elif probs is not None:
            pred_class = int(np.argmax(probs)) 
        else:
            pred_class = -1
        
        response = {"predicted_class": pred_class, "probabilities": probs}
        if embedding is not None:
            response["embedding"] = embedding
        
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 1. Accettiamo una lista di nomi separati da virgola
    parser.add_argument("--model_names", required=True, help="Lista di modelli separati da virgola (es: model1,model2)")
    
    # 2. Rileviamo il namespace (default automatico sul cluster se non specificato)
    parser.add_argument("--namespace", default="default", help="Namespace di Kubernetes in cui risiedono i Predictor")
    parser.add_argument("--predictor_port", type=int, default=8080)

    args, _ = parser.parse_known_args()

    # Parsiamo la stringa dei modelli in una lista Python pulita
    models_to_serve = [name.strip() for name in args.model_names.split(",") if name.strip()]
    
    if not models_to_serve:
        raise ValueError("La lista --model_names non contiene modelli validi.")

    # Creiamo un'istanza di ImageTransformer per ogni modello richiesto
    instances_list = []
    for model_name in models_to_serve:
        transformer_instance = ImageTransformer(
            name=model_name, 
            namespace=args.namespace, 
            predictor_port=args.predictor_port
        )
        instances_list.append(transformer_instance)

    logger.info(f"Avvio di ModelServer con {len(instances_list)} istanze di Transformer caricate.")
    
    # Passiamo la lista completa di istanze al ModelServer
    ModelServer().start(instances_list)
"""
transformer:
    containers:
    - name: kserve-container
      image: tomasconti02/image-transformer:v14   # Versione aggiornata
      command: ["python", "-m", "transformer"]
      args:
      - --model_names
      - "simple-cnn,simple-cnn-test"               # <--- Solo i nomi separati da virgola!
      - --namespace
      - "default"                                  # <--- Il namespace di riferimento

"""
