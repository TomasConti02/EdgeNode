#!/bin/bash

echo "Start."

while true; do

    echo "--- launch batch request ---"

    curl -sf -X POST \
        "http://localhost:8080/predict_batch_encoded?models=simple-cnn,simple-cnn-test" \
        -H "Host: image-api.default.example.com" \
        -H "Content-Type: application/octet-stream" \
        -H "X-Image-Sizes: 674,674" \
        --data-binary @batch.bin \
        > /dev/null


    if [ $? -eq 0 ]; then
        echo "Batch response received"
    else
        echo "Error in batch response"
    fi


    echo "..."

    sleep 0.5

done
