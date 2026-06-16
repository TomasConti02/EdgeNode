### Directories
    - ./model  --> models and transformers container image creation
    - ./model_storage --> yaml manifest for model storage
    - ./model_testing --> commands for inference testing
    - ./loadBalancer --> yaml manifest and configuration of a LoadBlancer
    
-------------------------------------------------------------------------
### Deploy by yaml manifests

Creation of the PVC and PV locally. Mount a volume with the inference model's weights
```bash
kubectl apply -f model_storage/single-pvc.yaml
```
```text
kubectl get pvc
NAME                  STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
simple-cnn-pvc        Bound    pvc-11acdcb1-e03a-43c4-add2-c07c069e50e9   100Mi      RWO            standard       <unset>                 3h27m
kubectl get pv
NAME                                       CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                         STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
pvc-11acdcb1-e03a-43c4-add2-c07c069e50e9   100Mi      RWO            Delete           Bound    default/simple-cnn-pvc        standard       <unset>                          3h28m

```
Deploy of the inference kserve model:
```bash
kubectl apply -f inference-single.yaml

```
Inference model load the weights form the local storage. 
```text
kubectl get pods
NAME                                                            READY   STATUS    RESTARTS   AGE
simple-cnn-predictor-00001-deployment-8467f6bb67-7xjvg          2/2     Running   0          3h25m
simple-cnn-transformer-00001-deployment-85c9f44cfd-gc74l        2/2     Running   0          3h25m

```

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

Same steps for the deploy of multi-pvc.yaml and inference-multi.yaml.



