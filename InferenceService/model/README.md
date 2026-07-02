### Creation of a Kserve Inference model

```bash
python3 model.py
ls -R model_repo/1
```
---------------------------------------------------------------------

Docker configuration of the inference image:
```bash
docker login
docker build -t <image model name >:vx -f Dockerfile.model .
docker tag <image model name>:vx <your docker hub user>/<image model name>:vx
docker push < your docker hub user>/<image model name>:vx
```
-----------------------------------------------------------------------

Docker configuration of the transformer image:
```bash
docker login
docker build -t image-transformer:vx -f Dockerfile.transformer .
docker tag image-transformer:vx <your docker hub user>/image-transformer:vx
docker push <your docker hub user>/image-transformer:vx
```
-----------------------------------------------------------------------

While multi-stage builds are effective at excluding build-time toolchains, the final image remains significant in size. This is due to the inherent weight of the TensorFlow library, which is a required dependency for the inference and transformation logic. Even in its optimized form, the runtime footprint includes the necessary core libraries to ensure model compatibility

----

### Kserve Inference model

KServe is an enterprise MLOps framework designed to deploy machine learning models as highly scalable microservices on Kubernetes. To maximize resource efficiency, KServe decouples data processing from heavy tensor computation by separating the workload into distinct components: a Transformer (handling pre- and post-processing) and a Predictor (dedicated solely to model inference).

Inside the Transformer container, a dual-engine processing model utilizes two distinct execution environments running side-by-side to guarantee low-latency request handling:

- Synchronous Engine (Main Request Threads): Managed natively by KServe's web server. Its primary responsibility is high-speed execution of data transformation algorithms (preprocess and postprocess) and making synchronous HTTP/gRPC routing calls directly to the model Predictor.

- Asynchronous Engine (Background Worker Thread): A dedicated, non-blocking background companion driven by a custom Python asyncio event loop. Its isolated priority is managing concurrent network ingestion and streaming event payloads out to the Knative/Kafka broker via CloudEvents, completely out of the critical path of the user request.

The main idea is to avoid Main Thread blocked because network call latency.