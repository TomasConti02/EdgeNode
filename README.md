## EdgeNode

EdgeNode is a Kubernetes-based AI inference platform designed to serve machine learning models into 5G k8s computing environments.

I'm testing deployment on my PC with limited resources.

For this reason there are some mock elements:

- CNN models 
- pre-processing, i have to figure out the right way to implement it
- there is not a load balancer for the istio gataway
- models weight are not loaded from a registry in cloud but are locally stored into pv

Kserve allow to:
- scale in-out inference services
- deploy workload into gpu-cpu
- define priority for the runtime scheduler 

---

## Architecture
All the project has been executed on top of a Linux distriution as ubuntu 24.04
EdgeNode is built on top of a modern Kubernetes stack:

- **KServe** → ML model serving and inference
- **Knative Serving** → autoscaling and serverless workloads
- **Istio** → service mesh for traffic management, security, and observability all by sidecar proxyes

---

## Prerequisites

Before getting started, make sure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/) 
- [kubectl](https://kubernetes.io/docs/tasks/tools/) (by snap is good as well)
- [Helm](https://helm.sh/docs/intro/install/) (by snap is good as well)

---

## Supported Cluster Types

EdgeNode can run on different Kubernetes environments:

### Kind (recommended for local development)
A lightweight Kubernetes cluster running entirely inside Docker containers.
- [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
### K3s on VMs
A lightweight Kubernetes distribution running on virtual machines managed by a hypervisor.
- [k3s](https://docs.k3s.io/quick-start)
---

## Quick Start

### Create a Kind cluster
The cluster will be created with:
- 1 control-plane node  
- 2 worker nodes
- 
```bash
kind create cluster --config kind-config.yaml
```

### Setup the Kserve cluster

The setup installs the following main components:
- Istio → service mesh for traffic management, security, and observability
- Knative → autoscaling and serverless workloads
- KServe → ML model serving and inference
- cert-manager → automatic TLS certificate management
- 
Start setup operation:

```bash
bash kserve/hack/quick_install.sh
```

## Istio Ingress Gateway Tunneling

Kind does not provide a cloud-native LoadBalancer. To bypass the <pending> status of the Istio gataway SVC create a TCP tunnel. This maps port 8080 on the host machine to port 80 on the Ingress Gateway pod where it is listening.

 ```bash
kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80
```

--------------------------------------------------------------------------------
### Fast deploy

There is a fast deployment script for the inference cluster on kserve.
into cont.json there is the cluster configuration used by the deployment script. 

```bash
cd ./Deployment
python3 deploy-multi.py 

usage: deploy-multi.py [-h] [--deploy] [--delete] [--namespace NAMESPACE] [--config CONFIG]
Deployment and Cleanup Script for KServe.
options:
  -h, --help            show this help message and exit
  --deploy              Executes model deployment.
  --delete              Executes model deletion.
  --namespace NAMESPACE
                        Target namespace (default: 'default').
  --config CONFIG       Path to config file.
```
Execute the deploy:
```bash
python3 deploy-multi.py  --deploy
```
Check the cluster state:
```bash
kubectl get pods
NAME                                                            READY   STATUS    RESTARTS   AGE
simple-cnn-predictor-00001-deployment-556cb9c54f-dsfkm          2/2     Running   0          5m1s
simple-cnn-test-predictor-00001-deployment-6b74788d97-cbr6k     2/2     Running   0          4m56s
simple-cnn-test-transformer-00001-deployment-85c874dc69-nhgwx   2/2     Running   0          4m56s
simple-cnn-transformer-00001-deployment-6c7df5f947-cjjgx        2/2     Running   0          5m1s
```
Check the kserve knative inference services avalilability:
```bash
kubectl get ksvc
NAME                          URL                                                      LATESTCREATED                       LATESTREADY                         READY   REASON
simple-cnn-predictor          http://simple-cnn-predictor.default.example.com          simple-cnn-predictor-00001          simple-cnn-predictor-00001          True    
simple-cnn-test-predictor     http://simple-cnn-test-predictor.default.example.com     simple-cnn-test-predictor-00001     simple-cnn-test-predictor-00001     True    
simple-cnn-test-transformer   http://simple-cnn-test-transformer.default.example.com   simple-cnn-test-transformer-00001   simple-cnn-test-transformer-00001   True    
simple-cnn-transformer        http://simple-cnn-transformer.default.example.com        simple-cnn-transformer-00001        simple-cnn-transformer-00001        True    
```

## Inference Service Test

Use 'localhost:8080' because the tunnel bridges the local interface to  the cluster network beacuse the port-forwarding.
```bash
curl -X POST http://localhost:8080/v1/models/simple-cnn-test:predict \ 
     -H "Host: simple-cnn-test.default.example.com" \
     -H "Content-Type: application/json" \
     -d @image.json
{
"predicted_class":0,
"probabilities"[0.109329447,0.0996421501,...,0.0744211152],
"embedding":[0.0178625677,0.132521331,..,0.0090905251,0.184597969]
}
```
Test both the models
```bash
curl -X POST http://localhost:8080/v1/models/simple-cnn:predict \dict \
     -H "Host: simple-cnn.default.example.com" \
     -H "Content-Type: application/json" \
     -d @image.json
{
"predicted_class":0,
"probabilities":[0.109329447,0.0996421501..2,0.102014825,0.0744211152],
"embedding":[0.0178625677,..,0.184597969]
}
```
Delate the inference cluster:
```bash
python3 deploy-multi.py --delete
==> Starting cleanup in namespace: 'default'

--- Cleaning resources for: simple-cnn ---
  [DELETED] InferenceService
  [DELETED] PVC

--- Cleaning resources for: simple-cnn-test ---
  [DELETED] InferenceService
  [DELETED] PVC
```
## Workflow (short)

1. **Client** → POST to the istio gataway `http://localhost:8080/v1/models/<model>:predict`  
   Header `Host: <model>.<namespace>.example.com` allow Istio the detect the target
  > - `localhost:8080` → the Istio ingress gateway is exposed locally via `kubectl port-forward` for testing.  
  > - `/v1/models/<model>:predict` → the standard KServe prediction endpoint, where `<model>` is the name of the InferenceService.  
  > - The gateway internally routes the request to the correct transformer/predictor based on the `Host` header.

3. **Istio Ingress Gateway** routes request to the **Transformer**  
   `{model}-transformer.{namespace}.svc.cluster.local`

4. **Transformer** preprocesses:  
   decode base64 → grayscale → resize 28×28 → normalize

5. **Transformer** calls **Predictor** internally  
   `{model}-predictor.{namespace}.svc.cluster.local:8080`

6. **Predictor** runs inference → returns raw predictions

7. **Transformer** postprocesses: extracts `predicted_class`, `probabilities`, `embedding`

8. **Gateway** returns final JSON response to client
![Demo Kiali Inference](sample/stream.gif)
-------------------------------------------------------------------------------

### Deploy by yaml manifests

The mock inference model is loaded localy into the cluster into a PV.
The directory mounting point is standard for kserve auto lookup and attach.
```bash
kubectl apply -f Model_local_storing_PVC/pvc.yaml
```
Into a fore advance version of the edge node the inference have to be loaded from a remote repository in cloud.

### Inference model and trasformer sidecar 
Deploy the inference model together with its preprocessing/transformer sidecar service using a KServe InferenceService.
```bash
kubectl apply -f InferenceService/inference2.yaml
```
```text
kubectl get inferenceservice
NAME         URL                                     READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION          AGE
simple-cnn   http://simple-cnn.default.example.com   True           100                              simple-cnn-predictor-00001   6m23s
```
```text
kubectl get pods
NAME                                                       READY   STATUS    RESTARTS   AGE
simple-cnn-predictor-00001-deployment-7b9c7f4b9-sf5wp      2/2     Running   0          6m43s
simple-cnn-transformer-00001-deployment-64576945cf-qj782   2/2     Running   0          6m43s
```
### Inference testing by Rest POST operation as :
You can test the deployed model using a REST POST request:
```bash
 curl -X POST http://localhost:8080/v1/models/simple-cnn:predict \
     -H "Host: simple-cnn.default.example.com" \
     -H "Content-Type: application/json" \
     -d @image.json

{
"predicted_class":0,
"probabilities":[0.109329447,.....,0.0744211152],
"embedding"[0.0178625677,0.132521331,....,0.0090905251,0.184597969]
}
```
In real applciation scenario this Rest operation is done by the mobile device asking for a heavy inference operation to the edge node passing the cropped image.


