### Directories
- ./model  --> models and transformers container images code
- ./model_testing --> commands for testing the inference pipeline
- ./loadBalancer --> yaml manifest and configuration of a LoadBlancer
    
-------------------------------------------------------------------------
### Deploy by yaml manifests

The first yaml manifest deploy on the cluster models weight into pvc-pv.
The second yaml manifest build up all the inference service based on these models weights.

```bash
kubectl apply -f 01-jobs-and-pvcs.yaml
kubectl apply -f 02-inferenceservices.yaml
```




