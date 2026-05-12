# Infrastructure Setup

A Kubernetes cluster is required to host the **Edge Node**. Choose the deployment method that best fits your environment:
*   **K3s**: Best for connecting **Hypervisor VMs**.
*   **KinD (Kubernetes in Docker)**: Best for **local development** using Docker containers as nodes.
---
## Create a KinD Cluster
To provision a cluster with **one control plane** and **two worker nodes**, use the provided configuration file:
Execute the following command in your terminal:
```bash
kind create cluster --config kind-config.yaml
```
to delate the k8s kind cluster exec:
```bash
kind delete cluster
```
to launch the kserve configuration and all the dependency:
```bash
bash kserve/hack/quick_install.sh
```
