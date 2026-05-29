import time
import argparse
from kubernetes import client, config

def wait_for_pvc_bound(core_v1_api, pvc_name, namespace, timeout=120, check_interval=3):
    """
    Blocks execution until the PVC changes its status to 'Bound' or a timeout occurs.
    """
    print(f"[WAIT] Waiting for PVC '{pvc_name}' to become Bound...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            pvc = core_v1_api.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
            if pvc.status.phase == "Bound":
                print(f"[READY] PVC '{pvc_name}' is Bound and ready.")
                return True
            print(f"[STATUS] PVC '{pvc_name}' current state: {pvc.status.phase}. Retrying in {check_interval}s...")
        except client.exceptions.ApiException as e:
            print(f"[WARNING] Error checking PVC: {e.reason}")
        
        time.sleep(check_interval)
        
    raise TimeoutError(f"Timeout reached! PVC '{pvc_name}' did not become Bound within {timeout} seconds.")


def deploy_models(models_list, namespace="default"):
    config.load_kube_config()
    
    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()
    custom_api = client.CustomObjectsApi()

    print(f"\n||| STARTING DEPLOYMENT PROCESS IN NAMESPACE '{namespace}' |||\n")

    for model_item in models_list:
        # Defensive Check: Ensure model_item is actually a dictionary
        if isinstance(model_item, str):
            print(f"[ERROR] Expected a dictionary but got string: '{model_item}'. Check your my_models list format!")
            continue

        model_name = model_item["name"]
        model_image = model_item["image"]
        storage_size = model_item.get("storage_size", "100Mi")
        transformer_image = model_item.get("transformer_image", "tomasconti02/image-transformer:v13")
        
        pvc_name = f"{model_name}-pvc"
        job_name = f"{model_name}-copy-job"
        
        print(f"=== Starting deployment for model: {model_name} ===")

        # -------------------------------------------------------------
        # 1. PVC CREATION
        # -------------------------------------------------------------
        pvc_manifest = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": pvc_name, "namespace": namespace},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {"requests": {"storage": storage_size}}
            }
        }
        
        try:
            core_v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc_manifest)
            print(f"[OK] PVC '{pvc_name}' created successfully.")
        except client.exceptions.ApiException as e:
            if e.status == 409:
                print(f"[INFO] PVC '{pvc_name}' already exists.")
            else:
                print(f"[ERROR] Error creating PVC: {e.reason}")
                continue

        # -------------------------------------------------------------
        # 2. JOB CREATION
        # -------------------------------------------------------------
        job_args = "mkdir -p /mnt/models/1;\ncp -r /app/model/* /mnt/models/1/;\nls -lh /mnt/models/1;\n"

        job_manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "namespace": namespace},
            "spec": {
                "ttlSecondsAfterFinished": 10,
                "backoffLimit": 1,
                "template": {
                    "spec": {
                        "restartPolicy": "OnFailure",
                        "containers": [{
                            "name": "model-writer",
                            "image": model_image,
                            "command": ["/bin/sh", "-c"],
                            "args": [job_args],
                            "volumeMounts": [{"name": "model-storage", "mountPath": "/mnt/models"}]
                        }],
                        "volumes": [{"name": "model-storage", "persistentVolumeClaim": {"claimName": pvc_name}}]
                    }
                }
            }
        }

        try:
            batch_v1.create_namespaced_job(namespace=namespace, body=job_manifest)
            print(f"[OK] Job '{job_name}' started.")
        except client.exceptions.ApiException as e:
            if e.status == 409:
                print(f"[INFO] Job '{job_name}' is already running or exists.")
            else:
                print(f"[ERROR] Could not create Job '{job_name}': {e.reason}")

        # -------------------------------------------------------------
        # 3. WAIT FOR PVC TO BIND
        # -------------------------------------------------------------
        try:
            wait_for_pvc_bound(core_v1, pvc_name, namespace)
        except TimeoutError as err:
            print(f"[CRITICAL] {err} Skipping KServe deployment for this model.")
            continue

        # -------------------------------------------------------------
        # 4. KSERVE INFERENCESERVICE CREATION
        # -------------------------------------------------------------
        predictor_host = f"{model_name}-predictor-00001.{namespace}.svc.cluster.local"

        isvc_manifest = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {"name": model_name, "namespace": namespace},
            "spec": {
                "predictor": {
                    "model": {
                        "modelFormat": {"name": "tensorflow"},
                        "storageUri": f"pvc://{pvc_name}/"
                    }
                },
                "transformer": {
                    "containers": [{
                        "name": "kserve-container",
                        "image": transformer_image,
                        "command": ["python", "-m", "transformer"],
                        "args": ["--model_name", model_name, "--predictor_host", predictor_host]
                    }]
                }
            }
        }

        isvc_exists = False
        try:
            custom_api.get_namespaced_custom_object(
                group="serving.kserve.io", version="v1beta1", namespace=namespace,
                plural="inferenceservices", name=model_name
            )
            isvc_exists = True
            print(f"[INFO] InferenceService '{model_name}' already exists. Skipping creation.")
        except client.exceptions.ApiException as e:
            if e.status != 404:
                print(f"[ERROR] Error while checking InferenceService '{model_name}': {e.reason}")
                isvc_exists = True

        if not isvc_exists:
            try:
                custom_api.create_namespaced_custom_object(
                    group="serving.kserve.io", version="v1beta1", namespace=namespace,
                    plural="inferenceservices", body=isvc_manifest
                )
                print(f"[OK] InferenceService '{model_name}' created successfully.")
            except client.exceptions.ApiException as e:
                print(f"[ERROR] Could not create InferenceService '{model_name}': {e.reason}")
                
        print(f"=== Finished deployment for {model_name} ===\n")


def delete_models(models_list, namespace="default"):
    """
    Cleanly removes all resources associated with the provided models.
    """
    config.load_kube_config()
    
    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()
    custom_api = client.CustomObjectsApi()

    print(f"\n||| STARTING DELETION PROCESS IN NAMESPACE '{namespace}' |||\n")

    for model_item in models_list:
        if isinstance(model_item, str):
            continue
            
        model_name = model_item["name"]
        pvc_name = f"{model_name}-pvc"
        job_name = f"{model_name}-copy-job"

        print(f"--- Deleting resources for model: {model_name} ---")

        try:
            custom_api.delete_namespaced_custom_object(
                group="serving.kserve.io", version="v1beta1", namespace=namespace,
                plural="inferenceservices", name=model_name
            )
            print(f"[DELETED] InferenceService '{model_name}' removed.")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                print(f"[INFO] InferenceService '{model_name}' not found.")
            else:
                print(f"[ERROR] Error removing InferenceService '{model_name}': {e.reason}")

        try:
            batch_v1.delete_namespaced_job(name=job_name, namespace=namespace, propagation_policy="Background")
            print(f"[DELETED] Job '{job_name}' removed.")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                print(f"[INFO] Job '{job_name}' not found.")
            else:
                print(f"[ERROR] Error removing Job '{job_name}': {e.reason}")

        try:
            core_v1.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
            print(f"[DELETED] PVC '{pvc_name}' removed.")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                print(f"[INFO] PVC '{pvc_name}' not found.")
            else:
                print(f"[ERROR] Error removing PVC '{pvc_name}': {e.reason}")
        
        print(f"--- Finished deletion for {model_name} ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment and Cleanup Script for Models on KServe/Kubernetes.")
    parser.add_argument("--deploy", action="store_true", help="Executes model deployment.")
    parser.add_argument("--delete", action="store_true", help="Executes model deletion.")
    parser.add_argument("--namespace", type=str, default="default", help="Target Kubernetes namespace (default: 'default').")
    
    args = parser.parse_args()

    # CRITICAL: This MUST be a structured list containing explicit, clear dictionary instances
    my_models = [
        {
            "name": "simple-cnn",
            "image": "tomasconti02/mnist-embedding-model:1",
            "storage_size": "100Mi"
        },
        {
            "name": "simple-cnn-test",
            "image": "tomasconti02/mnist-embedding-model:1",
            "storage_size": "200Mi"
        }
    ]

    if not args.deploy and not args.delete:
        parser.print_help()
        print("\n[!] Error: You must specify at least one action (--deploy or --delete).")
        exit(1)

    if args.deploy:
        deploy_models(my_models, namespace=args.namespace)

    if args.delete:
        delete_models(my_models, namespace=args.namespace)

"""
kubectl get inferenceservice
kubectl get routes
kubectl get svc
"""

"""
kubectl get inferenceservice
NAME              URL                                          READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION               AGE
simple-cnn        http://simple-cnn.default.example.com        True           100                              simple-cnn-predictor-00001        9m43s
simple-cnn-test   http://simple-cnn-test.default.example.com   True           100                              simple-cnn-test-predictor-00001   8m34s

kubectl get routes
NAME                          URL                                                      READY   REASON
simple-cnn-predictor          http://simple-cnn-predictor.default.example.com          True    
simple-cnn-test-predictor     http://simple-cnn-test-predictor.default.example.com     True    
simple-cnn-test-transformer   http://simple-cnn-test-transformer.default.example.com   True    
simple-cnn-transformer        http://simple-cnn-transformer.default.example.com        True    

kubectl get svc
NAME                                        TYPE           CLUSTER-IP      EXTERNAL-IP                                            PORT(S)                                     AGE
kubernetes                                  ClusterIP      10.96.0.1       <none>                                                 443/TCP                                     73m
simple-cnn                                  ExternalName   <none>          knative-local-gateway.istio-system.svc.cluster.local   <none>                                      10m
simple-cnn-predictor                        ExternalName   <none>          knative-local-gateway.istio-system.svc.cluster.local   80/TCP                                      10m
simple-cnn-predictor-00001                  ClusterIP      10.96.103.252   <none>                                                 80/TCP,443/TCP                              10m
simple-cnn-predictor-00001-private          ClusterIP      10.96.76.27     <none>                                                 80/TCP,443/TCP,9090/TCP,9091/TCP,8012/TCP   10m
simple-cnn-test                             ExternalName   <none>          knative-local-gateway.istio-system.svc.cluster.local   <none>                                      9m38s
simple-cnn-test-predictor                   ExternalName   <none>          knative-local-gateway.istio-system.svc.cluster.local   80/TCP                                      9m30s
simple-cnn-test-predictor-00001             ClusterIP      10.96.97.146    <none>                                                 80/TCP,443/TCP                              9m32s
simple-cnn-test-predictor-00001-private     ClusterIP      10.96.206.173   <none>                                                 80/TCP,443/TCP,9090/TCP,9091/TCP,8012/TCP   9m32s
simple-cnn-test-transformer                 ExternalName   <none>          knative-local-gateway.istio-system.svc.cluster.local   80/TCP                                      9m19s
simple-cnn-test-transformer-00001           ClusterIP      10.96.82.2      <none>                                                 80/TCP,443/TCP                              9m32s
simple-cnn-test-transformer-00001-private   ClusterIP      10.96.113.42    <none>                                                 80/TCP,443/TCP,9090/TCP,9091/TCP,8012/TCP   9m32s
simple-cnn-transformer                      ExternalName   <none>          knative-local-gateway.istio-system.svc.cluster.local   80/TCP                                      10m
simple-cnn-transformer-00001                ClusterIP      10.96.45.148    <none>                                                 80/TCP,443/TCP                              10m
simple-cnn-transformer-00001-private        ClusterIP      10.96.102.16    <none>                                                 80/TCP,443/TCP,9090/TCP,9091/TCP,8012/TCP   10m
(env) tomas@tomas-ThinkPad-T15-Gen-2i:~/Desktop/EdgeNode$ 

"""