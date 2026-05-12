# 🚀 EdgeNode

**EdgeNode** is a Kubernetes-based AI inference platform designed for serving machine learning models at scale.  
It leverages **KServe**, **Knative**, and **Istio** to provide a cloud-native, scalable, and production-ready model serving infrastructure.

---

## 🧠 Architecture

EdgeNode is built on top of a modern Kubernetes stack:

- **KServe** → ML model serving and inference
- **Knative** → autoscaling and serverless workloads
- **Istio** → service mesh for traffic management, security, and observability

---

## 📦 Prerequisites

Before getting started, make sure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/) (or another compatible container runtime)
- [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Helm](https://helm.sh/docs/intro/install/)

---

## 🧩 Supported Cluster Types

EdgeNode can run on different Kubernetes environments:

### 🐳 Kind (recommended for local development)
A lightweight Kubernetes cluster running entirely inside Docker containers.

### 🖥️ K3s on VMs
A lightweight Kubernetes distribution running on virtual machines managed by a hypervisor.

---

## 🚀 Quick Start

### Create a Kind cluster
The cluster will be created with:
- 1 control-plane node  
- 2 worker nodes  
```bash
kind create cluster --config kind-config.yaml
```
### Setup the Kserve cluster
The setup installs the following main components:
- Istio → service mesh for traffic management, security, and observability
- Knative → autoscaling and serverless workloads
- KServe → ML model serving and inference
- cert-manager → automatic TLS certificate management
Start setup operation:
```bash
bash kserve/hack/quick_install.sh
```
### Load the mock inference model
The mock inference model is loaded localy into the cluster into a PV.
The directory mounting point is standard for kserve auto lookup and attach.
```bash
kubectl apply -f Model_local_storing_PVC/pvc.yaml
```
### Inference model and trasformer sidecar 
Deploy the inference model together with its preprocessing/transformer sidecar service using a KServe InferenceService.
```bash
kubectl apply -f InferenceService/inference2.yaml
kubectl get inferenceservice
NAME         URL                                     READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION          AGE
simple-cnn   http://simple-cnn.default.example.com   True           100                              simple-cnn-predictor-00001   6m23s
kubectl get pods
NAME                                                       READY   STATUS    RESTARTS   AGE
simple-cnn-predictor-00001-deployment-7b9c7f4b9-sf5wp      2/2     Running   0          6m43s
simple-cnn-transformer-00001-deployment-64576945cf-qj782   2/2     Running   0          6m43s
```
### Inference testing by Rest POST operation as :
You can test the deployed model using a REST POST request:
```bash
POST http://localhost:8080/v1/models/simple-cnn:predict \
     -H "Host: simple-cnn.default.example.com" \
     -H "Content-Type: application/json" \
     -d @image.json
```


