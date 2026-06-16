import argparse
import base64

from typing import Any

import tensorflow as tf

from kserve import Model, ModelServer


class ImageTransformer(Model):

    def __init__(
        self,
        name: str,
        namespace: str,
        predictor_port: int = 8080
    ) -> None:

        super().__init__(name)

        self.ready: bool = True
        self.predictor_port: int = predictor_port

        self.predictor_host: str = (
            f"{name}-predictor.{namespace}.svc.cluster.local"
        )

    def preprocess(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None
    ) -> dict[str, list[list[float]]]:

        processed: list[list[float]] = []

        for instance in payload["instances"]:

            img_bytes: bytes = base64.b64decode(
                instance["image"]["b64"]
            )

            image: tf.Tensor = tf.image.decode_image(
                img_bytes,
                channels=3
            )

            image = tf.image.rgb_to_grayscale(image)

            image = tf.image.resize(
                image,
                [28, 28]
            )

            image = (
                tf.cast(image, tf.float32)
                / 255.0
            )

            processed.append(
                image.numpy().tolist()
            )

        return {
            "instances": processed
        }

    def postprocess(
        self,
        outputs: dict[str, Any],
        headers: dict[str, str] | None = None
    ) -> dict[str, Any]:

        prediction: dict[str, Any] = (
            outputs["predictions"][0]
        )

        return {
            "predicted_class": int(
                prediction["predicted_class"]
            ),
            "probabilities": prediction[
                "probabilities"
            ],
            "embedding": prediction[
                "embedding"
            ]
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_names",
        required=True
    )

    parser.add_argument(
        "--namespace",
        default="default"
    )

    parser.add_argument(
        "--predictor_port",
        type=int,
        default=8080
    )

    args, _ = parser.parse_known_args()

    models: list[str] = [
        name.strip()
        for name in args.model_names.split(",")
        if name.strip()
    ]

    ModelServer().start([
        ImageTransformer(
            name=model_name,
            namespace=args.namespace,
            predictor_port=args.predictor_port
        )
        for model_name in models
    ])
