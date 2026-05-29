import argparse
import base64
import logging
import os
import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str = None, predictor_port: int = 8080):
        super().__init__(name)
        self.ready = True
        self.predictor_host = predictor_host or os.environ.get("PREDICTOR_HOST")
        if self.predictor_host is None:
            raise ValueError("Predictor host must be specified")
        self.predictor_port = predictor_port or int(os.environ.get("PREDICTOR_PORT", "8080"))
        logger.info(f"Predictor host: {self.predictor_host}:{self.predictor_port}")

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
            image = tf.image.decode_image(img_bytes, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            image = tf.cast(image, tf.float32) / 255.0
            processed.append(image.numpy().tolist())
        return {"instances": processed}

    def postprocess(self, outputs, headers=None):
        # Gestisce sia dict che list
        if isinstance(outputs, list):
            # Se è una lista, cerca un dizionario all'interno
            for item in outputs:
                if isinstance(item, dict):
                    outputs = item
                    break
        
        # Estrai i dati (gestisce il wrapper 'predictions')
        if isinstance(outputs, dict):
            data = outputs.get("predictions", outputs)
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            probs = data.get("probabilities")
            embedding = data.get("embedding")
            pred_class = data.get("predicted_class")
        else:
            probs = embedding = pred_class = None
        
        # Converti in liste Python
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
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--predictor_host", default=None)
    parser.add_argument("--predictor_port", type=int, default=8080)
    args, _ = parser.parse_known_args()
    model = ImageTransformer(args.model_name, args.predictor_host, args.predictor_port)
    ModelServer().start([model])
"""
import argparse
import base64
import logging
import os
import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str = None, predictor_port: int = 8080):
        super().__init__(name)
        self.ready = True
        self.predictor_host = predictor_host or os.environ.get("PREDICTOR_HOST")
        if self.predictor_host is None:
            raise ValueError("Predictor host must be specified")
        self.predictor_port = predictor_port or int(os.environ.get("PREDICTOR_PORT", "8080"))
        logger.info(f"Predictor host: {self.predictor_host}:{self.predictor_port}")

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
            image = tf.image.decode_image(img_bytes, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            image = tf.cast(image, tf.float32) / 255.0
            processed.append(image.numpy().tolist())
        return {"instances": processed}

    def postprocess(self, outputs, headers=None):
        # Log completo per debug
        logger.info(f"="*50)
        logger.info(f"OUTPUTS TYPE: {type(outputs)}")
        logger.info(f"OUTPUTS KEYS (if dict): {outputs.keys() if isinstance(outputs, dict) else 'N/A'}")
        logger.info(f"FULL OUTPUTS: {outputs}")
        logger.info(f"="*50)
        
        # Inizializza
        probs = None
        embedding = None
        pred_class = None
        
        # Caso 1: outputs è una lista
        if isinstance(outputs, list):
            logger.info(f"Outputs is a list of length {len(outputs)}")
            for i, item in enumerate(outputs):
                logger.info(f"  Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
                # Se trova una lista di 10, sono le probabilità
                if isinstance(item, (list, np.ndarray)) and len(item) == 10:
                    probs = item
                # Se trova una lista di 64, è l'embedding
                elif isinstance(item, (list, np.ndarray)) and len(item) == 64:
                    embedding = item
                # Se trova un intero, è predicted_class
                elif isinstance(item, (int, np.integer)):
                    pred_class = int(item)
        
        # Caso 2: outputs è un dict
        elif isinstance(outputs, dict):
            logger.info(f"Outputs is a dict with keys: {list(outputs.keys())}")
            # Cerca direttamente
            probs = outputs.get("probabilities")
            embedding = outputs.get("embedding")
            pred_class = outputs.get("predicted_class")
            
            # Se non trova, cerca in "predictions"
            if probs is None and "predictions" in outputs:
                preds = outputs["predictions"]
                if isinstance(preds, dict):
                    probs = preds.get("probabilities")
                    embedding = preds.get("embedding")
                    pred_class = preds.get("predicted_class")
                elif isinstance(preds, list) and len(preds) >= 3:
                    probs, embedding, pred_class = preds[0], preds[1], preds[2]
        
        # Fallback: cerca ricorsivamente
        if probs is None or len(probs) != 10:
            logger.info("Searching recursively for probabilities...")
            probs = self._find_by_length(outputs, 10)
        if embedding is None:
            logger.info("Searching recursively for embedding (length 64)...")
            embedding = self._find_by_length(outputs, 64)
        
        # Se ancora niente probabilità, errore
        if probs is None:
            logger.error("Could not find probabilities!")
            return {"error": "no probabilities found"}
        
        # Converti probabilità
        if hasattr(probs, 'tolist'):
            probs = probs.tolist()
        elif isinstance(probs, np.ndarray):
            probs = probs.tolist()
        elif not isinstance(probs, list):
            probs = list(probs)
        
        # Predicted class
        if pred_class is None and probs:
            pred_class = int(np.argmax(probs))
        elif pred_class is not None:
            pred_class = int(pred_class)
        
        # Converti embedding
        embedding_list = None
        if embedding is not None:
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            elif isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            elif isinstance(embedding, list):
                embedding_list = embedding
            else:
                embedding_list = list(embedding) if hasattr(embedding, '__iter__') else [embedding]
            logger.info(f"Embedding extracted, length: {len(embedding_list)}")
        
        # Risposta finale
        response = {"predicted_class": pred_class, "probabilities": probs}
        if embedding_list:
            response["embedding"] = embedding_list
            logger.info(f"Including embedding in response (first 5 values): {embedding_list[:5]}")
        else:
            logger.warning("No embedding found in predictor response")
        
        return response
    
    def _find_by_length(self, obj, target_len, depth=0):
        if depth > 5:
            return None
        if isinstance(obj, dict):
            for v in obj.values():
                res = self._find_by_length(v, target_len, depth+1)
                if res is not None:
                    return res
        elif isinstance(obj, (list, tuple, np.ndarray)):
            if len(obj) == target_len:
                return obj
            for item in obj:
                res = self._find_by_length(item, target_len, depth+1)
                if res is not None:
                    return res
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--predictor_host", default=None)
    parser.add_argument("--predictor_port", type=int, default=8080)
    args, _ = parser.parse_known_args()
    model = ImageTransformer(args.model_name, args.predictor_host, args.predictor_port)
    ModelServer().start([model])
"""
"""
import argparse
import base64
import logging
import os
import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str = None, predictor_port: int = 8080):
        super().__init__(name)
        self.ready = True
        self.predictor_host = predictor_host or os.environ.get("PREDICTOR_HOST")
        if self.predictor_host is None:
            raise ValueError("Predictor host must be specified via --predictor_host or PREDICTOR_HOST env variable.")
        self.predictor_port = predictor_port or int(os.environ.get("PREDICTOR_PORT", "8080"))
        logger.info(f"Predictor host: {self.predictor_host}:{self.predictor_port}")

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

            image = tf.image.decode_image(img_bytes, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            image = tf.cast(image, tf.float32) / 255.0
            processed.append(image.numpy().tolist())  # Convert to Python list for JSON

        return {"instances": processed}

    def postprocess(self, outputs, headers=None):
       
        logger.info(f"Raw outputs: {outputs}")

        # Unwrap possible 'predictions' wrapper
        if isinstance(outputs, dict):
            data = outputs.get("predictions", outputs)
        else:
            data = outputs

        probs = None
        embedding = None
        predicted_class = None

        if isinstance(data, dict):
            probs = data.get("probabilities")
            embedding = data.get("embedding")
            predicted_class = data.get("predicted_class")

        # Fallback: search recursively for a list of 10 floats (probabilities)
        if probs is None or not isinstance(probs, (list, np.ndarray)) or len(probs) != 10:
            probs = self._find_10_list(outputs)

        if probs is None:
            logger.warning("Could not extract 10-class probabilities from response")
            return {"error": "no valid probabilities"}

        # Convert probabilities to list of floats
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()
        elif not isinstance(probs, list):
            probs = list(probs)

        # Compute predicted class if not provided by model
        if predicted_class is None:
            probs_array = np.array(probs)
            predicted_class = int(np.argmax(probs_array))
        else:
            predicted_class = int(predicted_class)

        # Format embedding (if present)
        if embedding is not None:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                embedding = list(embedding) if hasattr(embedding, '__iter__') else [embedding]

        # Build final response
        response = {
            "predicted_class": predicted_class,
            "probabilities": probs
        }
        if embedding is not None:
            response["embedding"] = embedding

        return response

    def _find_10_list(self, obj, depth=0):
       
        if depth > 5:
            return None
        if isinstance(obj, dict):
            for v in obj.values():
                res = self._find_10_list(v, depth + 1)
                if res:
                    return res
        elif isinstance(obj, list):
            if len(obj) == 10 and all(isinstance(x, (float, int, np.floating, np.integer)) for x in obj):
                return [float(x) for x in obj]
            for item in obj:
                res = self._find_10_list(item, depth + 1)
                if res:
                    return res
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--predictor_host", default=None)
    parser.add_argument("--predictor_port", type=int, default=8080)
    args, _ = parser.parse_known_args()
    model = ImageTransformer(args.model_name, args.predictor_host, args.predictor_port)
    ModelServer().start([model])
"""
"""
import argparse
import base64
import logging
import os
import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str = None, predictor_port: int = 8080):
        super().__init__(name)
        self.ready = True
        self.predictor_host = predictor_host or os.environ.get("PREDICTOR_HOST")
        if self.predictor_host is None:
            raise ValueError("Predictor host must be specified via --predictor_host or PREDICTOR_HOST env variable.")
        self.predictor_port = predictor_port or int(os.environ.get("PREDICTOR_PORT", "8080"))
        logger.info(f"Predictor host: {self.predictor_host}:{self.predictor_port}")

    def preprocess(self, payload, headers=None):
        # Estrai le istanze dal payload
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
            # Se l'istanza è una lista, prendi il primo elemento (immagine)
            if isinstance(instance, list):
                instance = instance[0]
            # Decodifica base64
            img_b64 = instance["image"]["b64"]
            img_bytes = base64.b64decode(img_b64)
            # Preprocessing
            image = tf.image.decode_image(img_bytes, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            image = tf.cast(image, tf.float32) / 255.0
            # Converti numpy array in lista per JSON serialization
            processed.append(image.numpy().tolist())
        return {"instances": processed}

    def postprocess(self, outputs, headers=None):
        # Log per debug (opzionale, utile per diagnosticare)
        logger.info(f"Raw outputs: {outputs}")

        # Naviga possibili annidamenti: alcuni KServe wrappers mettono tutto sotto "predictions"
        if isinstance(outputs, dict):
            # Cerca prima "predictions", poi usa outputs stesso
            data = outputs.get("predictions", outputs)
            # Se data è ancora un dict, prendi la lista "probabilities"
            if isinstance(data, dict):
                probs = data.get("probabilities")
            else:
                probs = None
        else:
            probs = None

        # Se non trovato, cerca ricorsivamente una lista di 10 float (fallback sicuro)
        if probs is None or not isinstance(probs, (list, np.ndarray)) or len(probs) != 10:
            probs = self._find_10_list(outputs)

        if probs is None:
            logger.warning("Could not extract 10-class probabilities from response")
            return {"predicted_class": -1, "error": "no valid probabilities"}

        # Converti in array numpy e calcola la classe con probabilità massima
        probs_array = np.array(probs)
        predicted_class = int(np.argmax(probs_array))

        # Restituisci una risposta pulita
        return {
            "predicted_class": predicted_class,
            "probabilities": probs_array.tolist()
        }

    def _find_10_list(self, obj, depth=0):
        #Ricerca ricorsiva di una lista di 10 float in strutture annidate.
        if depth > 5:
            return None
        if isinstance(obj, dict):
            for v in obj.values():
                res = self._find_10_list(v, depth + 1)
                if res:
                    return res
        elif isinstance(obj, list):
            if len(obj) == 10 and all(isinstance(x, (float, int, np.floating, np.integer)) for x in obj):
                return [float(x) for x in obj]
            for item in obj:
                res = self._find_10_list(item, depth + 1)
                if res:
                    return res
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--predictor_host", default=None)
    parser.add_argument("--predictor_port", type=int, default=8080)
    args, _ = parser.parse_known_args()
    model = ImageTransformer(args.model_name, args.predictor_host, args.predictor_port)
    ModelServer().start([model])
"""
"""
import argparse
import base64
import logging
import os
import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str = None, predictor_port: int = 8080):
        super().__init__(name)
        self.ready = True
        self.predictor_host = predictor_host or os.environ.get("PREDICTOR_HOST")
        if self.predictor_host is None:
            raise ValueError("Predictor host must be specified via --predictor_host or PREDICTOR_HOST env variable.")
        self.predictor_port = predictor_port or int(os.environ.get("PREDICTOR_PORT", "8080"))
        logger.info(f"Predictor host: {self.predictor_host}:{self.predictor_port}")

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
            image = tf.image.decode_image(img_bytes, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            image = tf.cast(image, tf.float32) / 255.0
            # CONVERSIONE A LISTA (fix JSON serialization)
            processed.append(image.numpy().tolist())
        return {"instances": processed}

    def postprocess(self, outputs, headers=None):
        logger.info(f"Outputs type: {type(outputs)}")
        logger.info(f"Outputs: {outputs}")

        probs = None
        if isinstance(outputs, dict):
            preds = outputs.get("predictions", outputs.get("outputs", outputs))
            if isinstance(preds, dict):
                probs = preds.get("probabilities", preds.get("logits", preds.get("scores")))
                if isinstance(probs, dict):
                    probs = probs.get("embedding") or next((v for v in probs.values() if isinstance(v, list)), None)
            elif isinstance(preds, list):
                probs = preds[0] if preds else None
            if probs is None:
                probs = outputs.get("probabilities")
                if isinstance(probs, dict):
                    probs = probs.get("embedding") or next((v for v in probs.values() if isinstance(v, list)), None)
        elif isinstance(outputs, list):
            probs = outputs[0] if outputs else None

        if probs is None:
            logger.warning("No probabilities extracted")
            return {"predicted_class": -1, "error": "no probabilities"}

        probs = np.array(probs)
        predicted_class = int(np.argmax(probs))
        return {"predicted_class": predicted_class, "probabilities": probs.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--predictor_host", default=None)
    parser.add_argument("--predictor_port", type=int, default=8080)
    args, _ = parser.parse_known_args()
    model = ImageTransformer(args.model_name, args.predictor_host, args.predictor_port)
    ModelServer().start([model])
"""

"""
import argparse
import base64
import json
import logging
import os
import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str = None, predictor_port: int = 8080):
        super().__init__(name)
        self.ready = True

        self.predictor_host = predictor_host or os.environ.get("PREDICTOR_HOST")
        if self.predictor_host is None:
            raise ValueError(
                "Predictor host must be specified via --predictor_host or PREDICTOR_HOST env variable."
            )

        self.predictor_port = predictor_port or int(os.environ.get("PREDICTOR_PORT", "8080"))

        logger.info(f"Predictor host: {self.predictor_host}")
        logger.info(f"Predictor port: {self.predictor_port}")

    def preprocess(self, payload, headers=None):
        #Decode base64 image and convert to model input tensor.

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

            image = tf.image.decode_image(img_bytes, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            image = tf.cast(image, tf.float32) / 255.0

            # Conversione a lista Python per serializzazione JSON
            processed.append(image.numpy().tolist())

        return {"instances": processed}

    def postprocess(self, outputs, headers=None):
        #Handle multi-output model response.
        probs = None

        if isinstance(outputs, dict):
            # Cerca in "predictions" o "outputs"
            preds = outputs.get("predictions", outputs.get("outputs"))
            if preds is not None:
                if isinstance(preds, dict):
                    probs = preds.get("probabilities")
                elif isinstance(preds, list):
                    probs = preds[0] if preds else None
            # Cerca direttamente "probabilities"
            if probs is None:
                probs = outputs.get("probabilities")
            # Se il dizionario ha un solo valore, prova quello
            if probs is None and len(outputs) == 1:
                maybe_probs = list(outputs.values())[0]
                if isinstance(maybe_probs, list):
                    probs = maybe_probs[0] if maybe_probs else None

        elif isinstance(outputs, list):
            # La risposta è direttamente una lista di probabilità
            probs = outputs[0] if outputs else None

        if probs is None:
            logger.warning("No probabilities found in response")
            return {"predicted_class": -1}

        probs = np.array(probs)
        predicted_class = int(np.argmax(probs))
        return {
            "predicted_class": predicted_class,
            "probabilities": probs.tolist()
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--predictor_host", default=None)
    parser.add_argument("--predictor_port", type=int, default=8080)

    args, _ = parser.parse_known_args()

    model = ImageTransformer(
        name=args.model_name,
        predictor_host=args.predictor_host,
        predictor_port=args.predictor_port
    )

    ModelServer().start([model])
"""
"""
import argparse
import base64
import json
import logging
import os
import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str = None, predictor_port: int = 8080):
        super().__init__(name)
        self.ready = True

        # 1. Use explicitly passed argument if provided
        if predictor_host:
            self.predictor_host = predictor_host
        # 2. Otherwise read from environment
        else:
            self.predictor_host = os.environ.get("PREDICTOR_HOST")

        # 3. If still None, raise an error (no hardcoded fallback)
        if self.predictor_host is None:
            raise ValueError(
                "Predictor host must be specified via --predictor_host argument "
                "or PREDICTOR_HOST environment variable."
            )

        logger.info(f"Predictor host: {self.predictor_host}")

        # Predictor port (default 8080, can be overridden)
        self.predictor_port = predictor_port or int(os.environ.get("PREDICTOR_PORT", "8080"))
        logger.info(f"Predictor port: {self.predictor_port}")

    def preprocess(self, payload, headers=None):
        #Extract base64 images, preprocess them, and return tensors for the predictor.
        logger.debug(f"Raw payload: {json.dumps(payload, indent=2)}")

        # Normalise payload format
        if isinstance(payload, list):
            instances = payload
        elif isinstance(payload, dict):
            instances = payload.get("instances", [])
        else:
            raise ValueError("Unsupported payload format: expected list or dict with 'instances'")

        if not instances:
            logger.warning("No instances found in payload")
            return {"instances": []}

        processed = []

        for idx, instance in enumerate(instances):
            # Handle case where instance is a list containing a single dict
            if isinstance(instance, list):
                if instance:
                    instance = instance[0]
                else:
                    logger.warning(f"Empty list at index {idx}, skipping")
                    continue

            if not isinstance(instance, dict):
                raise ValueError(f"Instance {idx} is not a dict: {type(instance)}")
            if "image" not in instance:
                raise KeyError(f"Instance {idx} missing 'image' key. Keys: {instance.keys()}")
            image_dict = instance["image"]
            if "b64" not in image_dict:
                raise KeyError(f"Instance {idx} missing 'b64' in image dict. Keys: {image_dict.keys()}")

            img_b64 = image_dict["b64"]
            img_bytes = base64.b64decode(img_b64)

            # Preprocess: decode, grayscale, resize, normalize
            image = tf.image.decode_image(img_bytes, channels=3)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            image = tf.cast(image, tf.float32) / 255.0

            processed.append({"input": image.numpy().tolist()})

        return {"instances": processed}

    def postprocess(self, outputs, headers=None):
        #Extract predictions and return the class label.
        preds = outputs.get("predictions")
        if preds is None or len(preds) == 0:
            logger.warning("No predictions received")
            return {"predicted_class": -1}

        predicted_class = int(np.argmax(preds[0]))
        logger.info(f"Predicted class: {predicted_class}")
        return {"predicted_class": predicted_class}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--predictor_host", default=None, help="Predictor service hostname")
    parser.add_argument("--predictor_port", type=int, default=8080, help="Predictor port")
    args, _ = parser.parse_known_args()

    model = ImageTransformer(
        name=args.model_name,
        predictor_host=args.predictor_host,
        predictor_port=args.predictor_port
    )
    ModelServer().start([model])
"""
