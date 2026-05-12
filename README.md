# 🚀 EdgeNode

**EdgeNode** is a Kubernetes‑based AI inference platform that leverages **KServe**, **Knative**, and **Istio** to serve machine learning models locally. This guide walks you through setting up a KinD cluster and deploying an inference service with a custom transformer.

## 📋 Prerequisites

Make sure the following tools are installed on your machine:

- [Docker](https://docs.docker.com/get-docker/) (or another container runtime)
- [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Helm](https://helm.sh/docs/intro/install/) (for installing KServe components)

> **Note:** The cluster will be created with 1 control plane and 2 worker nodes.

## 1️⃣ Create the KinD Cluster

Create a file named `kind-config.yaml`:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
