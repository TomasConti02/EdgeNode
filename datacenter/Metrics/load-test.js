//k6 run load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.1.0/index.js";

const inferenceDuration = new Trend('inference_processing_time');

export const options = {
  stages: [
    { duration: '30s', target: 5 },   // Warm-up 
      { duration: '1m', target: 5 },   // standard
      { duration: '1m', target: 10 },   // Stress test
      //{ duration: '1m', target: 10 },   // standard
     // { duration: '1m', target: 30 },   // Stress test  
    { duration: '30s', target: 0 },   // Cool-down
  ],
  thresholds: {
    'http_req_failed': ['rate<0.01'], // error th of 1%
    'inference_processing_time': ['p(99)<3000'],
  },
};

const binaryData = open('./batch.bin', 'b');

export default function () {
  const url = 'http://192.168.17.37:31978/predict_batch_encoded?models=simple-cnn,simple-cnn-test';
  const params = { headers: {  'Host': 'image-api.default.example.com', 'Content-Type': 'application/octet-stream', 'X-Image-Sizes': '674,674', },
    timeout: '10s', // Timeout no infity block of req
  };

  const res = http.post(url, binaryData, params);
  const success = check(res, {  'status is 200': (r) => r.status === 200,  'body has content': (r) => r.body && r.body.length > 0, });
  if (success) {
    inferenceDuration.add(res.timings.duration);
  }
  sleep(0.1);
}

export function handleSummary(data) {
  return {
    "report1.html": htmlReport(data),
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}
