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


