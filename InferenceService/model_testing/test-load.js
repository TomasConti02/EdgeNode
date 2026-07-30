import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.1.0/index.js";

const model1InferenceDuration = new Trend('model1_inference_processing_time');
const alternatingInferenceDuration = new Trend('alternating_inference_processing_time');
const batchInferenceDuration = new Trend('batch_inference_processing_time');

// 7m 30s is each test duration with 1min of the stop between each end point test 
export const options = {
  scenarios: {
    // single end point call only simple-cnn (0s -> 7m30s)
    test_model_1: {
      executor: 'ramping-vus',
      startVUs: 0,
      startTime: '0s',
      stages: [
        { duration: '1m', target: 5 },   // warm up 
        { duration: '2m', target: 15 },  
        { duration: '2m', target: 30 },  
        { duration: '1m', target: 30 },  // sustained peak at 30 VU for 1 minute
        { duration: '1m', target: 15 },  
        { duration: '30s', target: 0 },  
      ],
      exec: 'testModel1',
    },
    
    // single end point call but with the change of the target model simple-cnn & simple-cnn-test (start after 8m30s with 1min of stop)
    test_alternating: {
      executor: 'ramping-vus',
      startVUs: 0,
      startTime: '8m30s',
      stages: [
        { duration: '1m', target: 5 },   // warm up 
        { duration: '2m', target: 15 },  // target are Virtual Users
        { duration: '2m', target: 30 },  
        { duration: '1m', target: 30 },  // sustained peak at 30 VU for 1 minute
        { duration: '1m', target: 15 },  
        { duration: '30s', target: 0 },  
      ],
      exec: 'testAlternating',
    },

    // Batch endpoint (start after 17m0s)
    test_batch: {
      executor: 'ramping-vus',
      startVUs: 0,
      startTime: '17m0s',
      stages: [
        { duration: '1m', target: 5 },   // warm up 
        { duration: '2m', target: 15 },  
        { duration: '2m', target: 30 },  
        { duration: '1m', target: 30 },  // sustained peak at 30 VU for 1 minute
        { duration: '1m', target: 15 },  
        { duration: '30s', target: 0 },  
      ],
      exec: 'testBatchEndpoint',
    },
  },
  
  thresholds: {
    'http_req_failed': ['rate<0.001'],
    'model1_inference_processing_time': ['p(95)<4000', 'p(99)<5000'],
    'alternating_inference_processing_time': ['p(95)<4000', 'p(99)<5000'],
    'batch_inference_processing_time': ['p(95)<4000', 'p(99)<5000'],
  },
};

const singleImageData = open('./immagine.png', 'b');
const batchBinaryData = open('./batch.bin', 'b');

// single test 
export function testModel1() {
  const url = 'http://192.168.17.37:31978/predict_encoded?model=simple-cnn';
  
  const params = { 
    headers: {  
      'Host': 'image-api.default.example.com', 
      'Content-Type': 'image/png', 
    },
    timeout: '15s',
  };

  const res = http.post(url, singleImageData, params);
  
  const success = check(res, {  
    'status is 200': (r) => r.status === 200,  
    'body has predicted_class': (r) => {
      try {
        const json = r.json();
        return json && json.predicted_class !== undefined;
      } catch (e) {
        return false;
      }
    }, 
  });
  
  if (success) {
    model1InferenceDuration.add(res.timings.duration);
  }
  
  sleep(0.2);
}

// switching single call model endpoint simple-cnn & simple-cnn-test 
export function testAlternating() {
  const useModelTest = __ITER % 2 === 0;
  const modelName = useModelTest ? 'simple-cnn-test' : 'simple-cnn';
  
  const url = `http://192.168.17.37:31978/predict_encoded?model=${modelName}`;
  
  const params = { 
    headers: {  
      'Host': 'image-api.default.example.com', 
      'Content-Type': 'image/png', 
    },
    timeout: '15s',
  };

  const res = http.post(url, singleImageData, params);
  
  const success = check(res, {  
    'status is 200': (r) => r.status === 200,  
    'body has predicted_class': (r) => {
      try {
        const json = r.json();
        return json && json.predicted_class !== undefined;
      } catch (e) {
        return false;
      }
    }, 
  });
  
  if (success) {
    alternatingInferenceDuration.add(res.timings.duration);
  }
  
  sleep(0.2);
}

// Batch endpoint
export function testBatchEndpoint() {
  const url = 'http://192.168.17.37:31978/predict_batch_encoded?models=simple-cnn,simple-cnn-test';
  
  const params = { 
    headers: {  
      'Host': 'image-api.default.example.com', 
      'Content-Type': 'application/octet-stream', 
      'X-Image-Sizes': '674,674', 
    },
    timeout: '15s', 
  };

  const res = http.post(url, batchBinaryData, params);
  
  const success = check(res, {  
    'batch status is 200': (r) => r.status === 200,  
    'batch body has content': (r) => r.body && r.body.length > 0, 
  });
  
  if (success) {
    batchInferenceDuration.add(res.timings.duration);
  }
  
  sleep(0.2);
}

export function handleSummary(data) {
  return {
    "report-sequential_alternating.html": htmlReport(data),
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}
