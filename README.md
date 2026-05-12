````markdown
# EdgeNode

EdgeNode is a Kubernetes-based edge AI inference platform built on top of **KServe**, **Knative**, and **Istio**.  
The project enables local deployment and testing of ML inference services inside a lightweight Kubernetes environment.

---

# Architecture Overview

The infrastructure is composed of:

- **KinD (Kubernetes in Docker)** for local cluster deployment
- **Istio** for networking and ingress management
- **Knative Serving** for serverless inference scaling
- **KServe** for ML model serving
- **Persistent Volumes (PVC)** for local model storage
- **InferenceService** resources for exposing models

---

# Prerequisites

Before starting, ensure the following tools are installed:

- Docker
- kubectl
- kind
- Helm
- Git

Recommended versions:

| Tool | Version |
|---|---|
| Docker | 24+ |
| Kubernetes | v1.35 |
| Kind | latest |
| Helm | v3+ |

---

# Clone the Repository

```bash
git clone https://github.com/TomasConti02/EdgeNode.git
cd EdgeNode
````

---

# Create the Kubernetes Cluster

The project includes a predefined KinD cluster configuration with:

* 1 Control Plane node
* 2 Worker nodes

Create the cluster using:

```bash
kind create cluster --config kind-config.yaml
```

Verify the cluster:

```bash
kubectl cluster-info
kubectl get nodes
```

Expected output:

```bash
NAME                 STATUS   ROLES           AGE
kind-control-plane   Ready    control-plane
kind-worker          Ready    <none>
kind-worker2         Ready    <none>
```

---

# Delete the Cluster

To completely remove the local Kubernetes cluster:

```bash
kind delete cluster
```

---

# Install KServe and Dependencies

The project provides an automated installation script that installs:

* Gateway API CRDs
* Istio
* Cert-Manager
* Knative Serving
* KServe

Run:

```bash
bash kserve/hack/quick_install.sh
```

The installation may take several minutes.

---

# Verify the Installation

Check all running pods:

```bash
kubectl get pods -A
```

Important namespaces:

* `istio-system`
* `knative-serving`
* `cert-manager`
* `kserve`

All pods should eventually reach:

```bash
STATUS = Running
```

---

# Common Installation Issue

If you encounter:

```bash
ImagePullBackOff
```

or:

```bash
429 Too Many Requests
```

Docker Hub rate limiting is blocking image downloads.

Fix:

```bash
docker login
```

Then preload the image into KinD:

```bash
docker pull kserve/kserve-controller:v0.17.0-rc1
kind load docker-image kserve/kserve-controller:v0.17.0-rc1
```

---

# Create Persistent Volume Claim

Before deploying the model, create the PVC used for local model storage:

```bash
kubectl apply -f ./Model_local_storing_PVC/pvc.yaml
```

Verify:

```bash
kubectl get pvc
```

---

# Deploy the Inference Service

Deploy the KServe inference service and transformer:

```bash
kubectl apply -f InferenceService/inference2.yaml
```

Verify the service:

```bash
kubectl get inferenceservices
```

Expected output:

```bash
NAME         URL                                        READY
simple-cnn   http://simple-cnn.default.example.com      True
```

---

# Enable Local Access with Port Forwarding

Expose the Istio ingress gateway locally:

```bash
kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80
```

The inference endpoint will now be reachable on:

```bash
http://localhost:8080
```

---

# Test Model Inference

Move into the testing directory:

```bash
cd InferenceService/model_testing
```

Execute the inference request:

```bash
curl -X POST http://localhost:8080/v1/models/simple-cnn:predict \
     -H "Host: simple-cnn.default.example.com" \
     -H "Content-Type: application/json" \
     -d @image.json
```

---

# Expected Response

The service should return a JSON prediction result similar to:

```json
{
  "predictions": [...]
}
```

---

# Useful Debug Commands

Check all resources:

```bash
kubectl get all -A
```

Inspect inference services:

```bash
kubectl get inferenceservices
```

Inspect pods:

```bash
kubectl get pods -A
```

Describe failing pods:

```bash
kubectl describe pod <pod-name> -n <namespace>
```

View logs:

```bash
kubectl logs <pod-name> -n <namespace>
```

---

# Project Structure

```bash
EdgeNode/
├── Deployment/
├── InferenceService/
├── Model_local_storing_PVC/
├── kserve/
├── kind-config.yaml
└── README.md
```

---

# Technologies Used

* Kubernetes
* KinD
* KServe
* Knative
* Istio
* Docker
* Helm

---

# Future Improvements

* GPU scheduling support
* Multi-model serving
* Edge-to-cloud synchronization
* Automated CI/CD deployment
* Monitoring and observability stack

---

# Author

Developed by Tomas Conti.

```
```
