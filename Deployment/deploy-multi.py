import time
import argparse
import json
import os
from kubernetes import client, config

def wait_for_pvc_bound(core_v1_api, pvc_name, namespace, timeout=120, check_interval=3):
    start_time = time.time()
    last_phase = None
    while time.time() - start_time < timeout: #loop to check if the pvc are bounded before launch the inference models deploy
        try:
            pvc = core_v1_api.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace) #use the k8s api to the the cluster pvc state
            current_phase = pvc.status.phase
            if current_phase == "Bound":
                print(f"  [PVC] '{pvc_name}' is Bound.")
                return True
            if current_phase != last_phase:
                print(f"  [PVC] '{pvc_name}' status: {current_phase}...")
                last_phase = current_phase
        except client.exceptions.ApiException as e:
            print(f"  [WARNING] Error checking PVC: {e.reason}")
        time.sleep(check_interval)
    raise TimeoutError(f"Timeout reached! PVC '{pvc_name}' did not become Bound.")


def deploy_models(  models_list,  namespace="default",  kubeconfig_path=None,  broker=None,  broker_host=None,  ce_type=None,  predictor_port=8080, ):

    if kubeconfig_path:
        config.load_kube_config(config_file=kubeconfig_path) #path for a k8s cluster
    else:
        config.load_kube_config() #local path for k8s cluster (kind)
    
    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()
    custom_api = client.CustomObjectsApi()

    print(f"==> Starting deployment in namespace: '{namespace}'")
    all_model_names = ",".join([m["name"] for m in models_list if isinstance(m, dict)]) #parsing the models json conf 
    print(f"==> Models to deploy: {all_model_names}\n")

    for model_item in models_list:
        if isinstance(model_item, str):
            print(f"[ERROR] Invalid format for item: '{model_item}'")
            continue
        #model feature extraction
        model_name = model_item["name"]
        model_image = model_item["image"]
        storage_size = model_item.get("storage_size", "100Mi")
        transformer_image = model_item.get("transformer_image", "tomasconti02/image-transformer:v14")
        
        pvc_name = f"{model_name}-pvc"
        job_name = f"{model_name}-copy-job"
        
        print(f"--- Deploying model: {model_name} ---")

        # local PVC creation for the modle weights and local loading
        pvc_manifest = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": pvc_name, "namespace": namespace},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {"requests": {"storage": storage_size}}
            }
        }
        try: #pvc create the req PV and used the deafult of define storage provider (default here )
            core_v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc_manifest)
            print(f"  [OK] PVC '{pvc_name}' created.")
        except client.exceptions.ApiException as e:
            if e.status == 409:
                print(f"  [INFO] PVC '{pvc_name}' already exists.")
            else:
                print(f"  [ERROR] PVC creation failed: {e.reason}")
                continue

        # Create a job that take the model weight and load into the pv
        job_args = "mkdir -p /mnt/models/1;\ncp -r /app/model/* /mnt/models/1/;\n"
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
            print(f"  [OK] Job '{job_name}' started.")
        except client.exceptions.ApiException as e:
            if e.status == 409:
                print(f"  [INFO] Job '{job_name}' already exists.")
            else:
                print(f"  [ERROR] Job creation failed: {e.reason}")

        # before deploy the inference service we need the pvc bounded 
        try:
            wait_for_pvc_bound(core_v1, pvc_name, namespace)
        except TimeoutError as err:
            print(f"  [CRITICAL] {err} Skipping KServe setup.")
            continue

        #  InferenceService CRD initialization
        isvc_manifest = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {"name": model_name, "namespace": namespace},
            "spec": {
                "predictor": {
                    "model": { 
                        "modelFormat": {"name": "tensorflow"}, #inference runtime
                        "storageUri": f"pvc://{pvc_name}/" #kserve have to check the model weights into a pvc
                    }
                },
                "transformer": {
                    "containers": [{
                        "name": "kserve-container",
                        "image": transformer_image,
                        "command": ["python", "-m", "transformer"], # at the container boot exec the transformer
                        "args": [
                            "--model_names", all_model_names,
                            "--namespace", namespace,
                            "--predictor_port", str(predictor_port),
                            "--broker", broker,
                            "--broker_host", broker_host,
                            "--ce_type", ce_type]
                    }]
                }
            }
        }
        # i do not want duplicates
        isvc_exists = False
        try: #check if the inferenceservices already exist
            custom_api.get_namespaced_custom_object( group="serving.kserve.io", version="v1beta1", namespace=namespace,  plural="inferenceservices", name=model_name )
            isvc_exists = True
            print(f"  [INFO] InferenceService '{model_name}' already exists.")
        except client.exceptions.ApiException as e:
            if e.status != 404:
                print(f"  [ERROR] Failed to check InferenceService: {e.reason}")
                isvc_exists = True

        if not isvc_exists:
            try: # create the CRD custom object calling the api on kube api custom
                custom_api.create_namespaced_custom_object(  group="serving.kserve.io", version="v1beta1", namespace=namespace, plural="inferenceservices", body=isvc_manifest )
                print(f"  [OK] InferenceService '{model_name}' created.")
            except client.exceptions.ApiException as e:
                print(f"  [ERROR] InferenceService creation failed: {e.reason}")
                
        print(f"Done with {model_name}.\n")


def delete_models(models_list, namespace="default", kubeconfig_path=None):
    if kubeconfig_path:
        config.load_kube_config(config_file=kubeconfig_path)
    else:
        config.load_kube_config() #from the load kube conf file, file for kind cluster

    core_v1 = client.CoreV1Api()
    batch_v1 = client.BatchV1Api()
    custom_api = client.CustomObjectsApi()
    
    print(f"==> Starting cleanup in namespace: '{namespace}'\n")
    for model_item in models_list:
        if isinstance(model_item, str):
            continue
        model_name = model_item["name"]
        pvc_name = f"{model_name}-pvc"
        job_name = f"{model_name}-copy-job"
        print(f"--- Cleaning resources for: {model_name} ---") #delate all model item resources
        try:
            custom_api.delete_namespaced_custom_object( group="serving.kserve.io", version="v1beta1", namespace=namespace, plural="inferenceservices", name=model_name )
            print(f"  [DELETED] InferenceService")
        except client.exceptions.ApiException as e:
            if e.status != 404:
                print(f"  [ERROR] Removing InferenceService: {e.reason}")

        try:
            batch_v1.delete_namespaced_job(name=job_name, namespace=namespace, propagation_policy="Background")
            print(f"  [DELETED] Job")
        except client.exceptions.ApiException as e:
            if e.status != 404:
                print(f"  [ERROR] Removing Job: {e.reason}")

        try:
            core_v1.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
            print(f"  [DELETED] PVC")
        except client.exceptions.ApiException as e:
            if e.status != 404:
                print(f"  [ERROR] Removing PVC: {e.reason}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment and Cleanup Script for KServe.")
    parser.add_argument("--deploy", action="store_true", help="Executes model deployment.")
    parser.add_argument("--delete", action="store_true", help="Executes model deletion.")
    parser.add_argument("--namespace", type=str, default="default", help="Target namespace (default: 'default').")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file.")
    args = parser.parse_args()

    if not args.deploy and not args.delete:
        parser.print_help()
        exit(1)

    if not os.path.exists(args.config): #check if the deploy conf file json exsist on this dir
        print(f"ERROR: '{args.config}' not found.")
        exit(1)
        
    try:
        with open(args.config, 'r') as f: #open the json descriprion file for the deployment configuration
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        exit(1)

    my_models = config_data.get("models", []) #load the models setting into the conf file 
    kubeconfig_path = config_data.get("kubeconfig_path", None) #path for the kube conf file 
    broker = config_data.get( "broker", "http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker" )
    broker_host = config_data.get( "broker_host", "kafka-broker-ingress.knative-eventing.svc.cluster.local" )
    ce_type = config_data.get( "ce_type", "org.kubeflow.serving.inference.request" )

    predictor_port = config_data.get("predictor_port", 8080)
    if args.deploy:
        deploy_models( my_models, namespace=args.namespace, kubeconfig_path=kubeconfig_path, broker=broker, broker_host=broker_host, ce_type=ce_type, predictor_port=predictor_port )

    if args.delete:
        delete_models(my_models, namespace=args.namespace, kubeconfig_path=kubeconfig_path)
