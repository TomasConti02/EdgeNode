import argparse
import base64
import logging
import os

import numpy as np
import tensorflow as tf
from kserve import Model, ModelServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


class ImageTransformer(Model):

    def __init__(
        self,
        name: str,
        namespace: str,
        predictor_port: int = 8080
    ):
        super().__init__(name)

        logger.info("=" * 80)
        logger.info("INITIALIZING IMAGE TRANSFORMER")
        logger.info(f"Model Name      : {name}")
        logger.info(f"Namespace       : {namespace}")
        logger.info(f"Predictor Port  : {predictor_port}")
        logger.info("=" * 80)

        self.ready = True
        self.predictor_port = predictor_port

        self.predictor_host = (
            f"{name}-predictor.{namespace}.svc.cluster.local"
        )

        logger.info(
            f"Predictor Host resolved as "
            f"{self.predictor_host}:{self.predictor_port}"
        )

    def preprocess(self, payload, headers=None):

        logger.info("")
        logger.info("=" * 80)
        logger.info("PREPROCESS START")
        logger.info("=" * 80)

        try:

            logger.info(f"Payload Type: {type(payload)}")

            if headers:
                logger.info(f"Headers: {headers}")

            if isinstance(payload, dict):
                logger.info(
                    f"Payload Keys: {list(payload.keys())}"
                )

            if isinstance(payload, list):
                instances = payload

            elif isinstance(payload, dict):
                instances = payload.get("instances", [])

            else:
                logger.error(
                    f"Unsupported payload type: {type(payload)}"
                )
                raise ValueError("Unsupported payload format")

            logger.info(
                f"Instances received: {len(instances)}"
            )

            if not instances:
                logger.warning("No instances found")
                return {"instances": []}

            processed = []

            for idx, instance in enumerate(instances):

                logger.info("-" * 80)
                logger.info(
                    f"Processing instance #{idx}"
                )
                logger.info("-" * 80)

                if isinstance(instance, list):
                    logger.info(
                        "Instance is list. Taking first element."
                    )
                    instance = instance[0]

                logger.info(
                    f"Instance keys: {list(instance.keys())}"
                )

                img_b64 = instance["image"]["b64"]

                logger.info(
                    f"Base64 length: {len(img_b64)}"
                )

                img_bytes = base64.b64decode(img_b64)

                logger.info(
                    f"Decoded bytes length: {len(img_bytes)}"
                )

                image = tf.image.decode_image(
                    img_bytes,
                    channels=3
                )

                logger.info(
                    f"Decoded image shape: {image.shape}"
                )

                image = tf.image.rgb_to_grayscale(
                    image
                )

                logger.info(
                    f"Grayscale shape: {image.shape}"
                )

                image = tf.image.resize(
                    image,
                    [28, 28]
                )

                logger.info(
                    f"Resized shape: {image.shape}"
                )

                image = tf.cast(
                    image,
                    tf.float32
                ) / 255.0

                logger.info(
                    f"Image dtype: {image.dtype}"
                )

                logger.info(
                    f"Image min value: "
                    f"{tf.reduce_min(image).numpy()}"
                )

                logger.info(
                    f"Image max value: "
                    f"{tf.reduce_max(image).numpy()}"
                )

                logger.info(
                    f"Final tensor shape: {image.shape}"
                )

                processed.append(
                    image.numpy().tolist()
                )

            logger.info(
                f"Preprocess completed. "
                f"Processed instances: {len(processed)}"
            )

            logger.info("=" * 80)
            logger.info("PREPROCESS END")
            logger.info("=" * 80)

            return {"instances": processed}

        except Exception as e:

            logger.exception(
                f"PREPROCESS ERROR: {str(e)}"
            )
            raise

    def postprocess(self, outputs, headers=None):

        logger.info("")
        logger.info("=" * 80)
        logger.info("POSTPROCESS START")
        logger.info("=" * 80)

        try:

            logger.info(
                f"Outputs Type: {type(outputs)}"
            )

            if headers:
                logger.info(
                    f"Headers: {headers}"
                )

            logger.info(
                f"Raw Outputs: {str(outputs)[:5000]}"
            )

            if isinstance(outputs, list):

                logger.info(
                    f"Outputs is list with "
                    f"{len(outputs)} elements"
                )

                for item in outputs:
                    if isinstance(item, dict):
                        outputs = item
                        break

            if isinstance(outputs, dict):

                logger.info(
                    f"Output keys: "
                    f"{list(outputs.keys())}"
                )

                data = outputs.get(
                    "predictions",
                    outputs
                )

                if isinstance(data, list):

                    logger.info(
                        f"Predictions list length: "
                        f"{len(data)}"
                    )

                    if len(data) > 0:
                        data = data[0]

                if isinstance(data, dict):
                    logger.info(
                        f"Prediction keys: "
                        f"{list(data.keys())}"
                    )

                probs = data.get("probabilities")
                embedding = data.get("embedding")
                pred_class = data.get(
                    "predicted_class"
                )

            else:

                logger.warning(
                    "Output is not a dictionary"
                )

                probs = None
                embedding = None
                pred_class = None

            logger.info(
                f"predicted_class: {pred_class}"
            )

            if probs is not None:

                logger.info(
                    f"Probabilities type: "
                    f"{type(probs)}"
                )

                if hasattr(probs, "__len__"):
                    logger.info(
                        f"Probabilities length: "
                        f"{len(probs)}"
                    )

            if embedding is not None:

                logger.info(
                    f"Embedding type: "
                    f"{type(embedding)}"
                )

                if hasattr(embedding, "__len__"):
                    logger.info(
                        f"Embedding length: "
                        f"{len(embedding)}"
                    )

            if probs is not None:

                if hasattr(probs, "tolist"):
                    probs = probs.tolist()

                elif isinstance(probs, np.ndarray):
                    probs = probs.tolist()

            if embedding is not None:

                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()

                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

            if pred_class is not None:

                pred_class = int(pred_class)

            elif probs is not None:

                pred_class = int(
                    np.argmax(probs)
                )

                logger.info(
                    f"Predicted class generated "
                    f"from argmax: {pred_class}"
                )

            else:

                pred_class = -1

                logger.warning(
                    "Unable to determine class."
                )

            response = {
                "predicted_class": pred_class,
                "probabilities": probs
            }

            if embedding is not None:
                response["embedding"] = embedding

            logger.info(
                f"Final response: "
                f"{str(response)[:5000]}"
            )

            logger.info("=" * 80)
            logger.info("POSTPROCESS END")
            logger.info("=" * 80)

            return response

        except Exception as e:

            logger.exception(
                f"POSTPROCESS ERROR: {str(e)}"
            )
            raise


if __name__ == "__main__":

    logger.info("")
    logger.info("=" * 80)
    logger.info("TRANSFORMER SERVER BOOT")
    logger.info("=" * 80)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_names",
        required=True,
        help="Comma separated model names"
    )

    parser.add_argument(
        "--namespace",
        default="default",
        help="Kubernetes namespace"
    )

    parser.add_argument(
        "--predictor_port",
        type=int,
        default=8080
    )

    args, _ = parser.parse_known_args()

    logger.info(
        f"Raw model_names argument: "
        f"{args.model_names}"
    )

    logger.info(
        f"Namespace: {args.namespace}"
    )

    logger.info(
        f"Predictor Port: "
        f"{args.predictor_port}"
    )

    models_to_serve = [
        name.strip()
        for name in args.model_names.split(",")
        if name.strip()
    ]

    logger.info(
        f"Parsed models: {models_to_serve}"
    )

    if not models_to_serve:

        logger.error(
            "No valid models found"
        )

        raise ValueError(
            "The --model_names list "
            "contains no valid models."
        )

    instances_list = []

    for model_name in models_to_serve:

        logger.info(
            f"Creating transformer for model: "
            f"{model_name}"
        )

        transformer = ImageTransformer(
            name=model_name,
            namespace=args.namespace,
            predictor_port=args.predictor_port
        )

        instances_list.append(
            transformer
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        f"Starting ModelServer with "
        f"{len(instances_list)} transformers"
    )
    logger.info("=" * 80)

    for model in instances_list:
        logger.info(
            f"Registered transformer: "
            f"{model.name}"
        )

    ModelServer().start(instances_list)