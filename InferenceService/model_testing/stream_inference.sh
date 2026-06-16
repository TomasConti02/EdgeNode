#!/bin/bash

echo "Start."

while true; do
    echo "--- launch req to simple-cnn ---"
    
    #curl -s -X POST http://localhost:8080/v1/models/simple-cnn:predict \
    curl -s -X POST http://172.18.255.200/v1/models/simple-cnn:predict \
         -H "Host: simple-cnn.default.example.com" \
         -H "Content-Type: application/json" \
         -d @image.json > /dev/null

    if [ $? -eq 0 ]; then
        echo "simple-cnn response received"
    else
        echo "Error in simple-cnn response"
    fi

    echo -e "\n--- launch req to simple-cnn-test ---"
    #curl -s -X POST http://localhost:8080/v1/models/simple-cnn-test:predict \
    curl -s -X POST http://172.18.255.200/v1/models/simple-cnn-test:predict \
         -H "Host: simple-cnn-test.default.example.com" \
         -H "Content-Type: application/json" \
         -d @image.json > /dev/null

    if [ $? -eq 0 ]; then
        echo "simple-cnn-test response received"
    else
        echo "Error in simple-cnn-test response"
    fi

    echo -e "\n..."
    sleep 2
done
