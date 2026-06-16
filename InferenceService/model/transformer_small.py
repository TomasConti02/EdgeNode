import argparse
import base64
from typing import Any, Dict, List, Optional

import tensorflow as tf
from kserve import Model, ModelServer


class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str, predictor_port: int = 8080) -> None:
        super().__init__(name)
        self.ready = True
        self.predictor_host = f"{predictor_host}:{predictor_port}"

    def preprocess(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[List[float]]]:
        processed = []
        for inst in payload["instances"]:
            img = tf.image.decode_image(base64.b64decode(inst["image"]["b64"]), channels=3)
            img = tf.image.rgb_to_grayscale(img)
            img = tf.image.resize(img, [28, 28])
            img = tf.cast(img, tf.float32) / 255.0
            processed.append(img.numpy().tolist())
        return {"instances": processed}

    def postprocess(
        self, outputs: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        pred = outputs["predictions"][0]
        return {
            "predicted_class": int(pred["predicted_class"]),
            "probabilities": pred["probabilities"],
            "embedding": pred["embedding"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Single model name")
    parser.add_argument("--model_names", type=str, help="Comma-separated model names")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--predictor_host", type=str, help="Full predictor host (overrides namespace)")
    parser.add_argument("--predictor_port", type=int, default=8080)
    args = parser.parse_known_args()[0]

    if args.model_name:
        models = [args.model_name]
    elif args.model_names:
        models = [n.strip() for n in args.model_names.split(",") if n.strip()]
    else:
        raise ValueError("Either --model_name or --model_names required")

    transformers = []
    for name in models:
        if args.predictor_host:
            host = args.predictor_host
        else:
            host = f"{name}-predictor.{args.namespace}.svc.cluster.local"
        transformers.append(ImageTransformer(name, host, args.predictor_port))

    ModelServer().start(transformers)