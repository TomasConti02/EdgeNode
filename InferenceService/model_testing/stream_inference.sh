#!/bin/bash

echo "Start."

while true; do
    echo "--- launch req to simple-cnn ---"
    curl -s -X POST http://localhost:8080/v1/models/simple-cnn:predict \
         -H "Host: simple-cnn.default.example.com" \
         -H "Content-Type: application/json" \
         -d @image.json

    echo -e "\n--- launch req to simple-cnn-test ---"
    curl -s -X POST http://localhost:8080/v1/models/simple-cnn-test:predict \
         -H "Host: simple-cnn-test.default.example.com" \
         -H "Content-Type: application/json" \
         -d @image.json

    echo -e "\n..."
    sleep 2
done
