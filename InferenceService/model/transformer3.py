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
        """Extract base64 images, preprocess them, and return tensors for the predictor."""
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
        """Extract predictions and return the class label."""
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