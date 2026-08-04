import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.1.0/index.js";

const inferenceDuration = new Trend('inference_processing_time');

export const options = {
  stages: [
    { duration: '1m', target: 5 },
    { duration: '2m', target: 15 },
    { duration: '3m', target: 30 },
    { duration: '3m', target: 30 },
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    'http_req_failed': ['rate<0.01'],
    'inference_processing_time': ['p(95)<4000', 'p(99)<5000'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
};

const imageData = open('./immagine.png', 'b');

export function testModel(modelName) {
  // Aggiornato con le porte NodePort corrette (30090 e 30091)
  const port = modelName === 'simple-cnn-test' ? '30091' : '30090';
  const url = `http://192.168.17.37:${port}/v1/models/${modelName}:predict`;
  
  const params = {
    headers: {
      'Content-Type': 'application/octet-stream',
    },
    timeout: '15s',
  };

  const res = http.post(url, imageData, params);
  
  const success = check(res, {
    'status is 200': (r) => r.status === 200,
    'has predicted_class': (r) => {
      try {
        const json = r.json();
        return json && json.predicted_class !== undefined;
      } catch (e) {
        return false;
      }
    },
  });

  if (success) {
    inferenceDuration.add(res.timings.duration);
  }

  sleep(0.2);
}

export default function() {
  const modelName = __ITER % 2 === 0 ? 'simple-cnn' : 'simple-cnn-test';
  testModel(modelName);
}

export function handleSummary(data) {
  return {
    "report-simple-cnn-test.html": htmlReport(data),
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}
