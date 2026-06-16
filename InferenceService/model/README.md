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
