import http from 'k6/http';
import { check } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.1.0/index.js";

const singleLatency = new Trend('single_inference_latency');
const alternatingLatency = new Trend('alternating_inference_latency');
const batchLatency = new Trend('batch_inference_latency');
const failedRequests = new Rate('inference_fail_rate');
const statusErrors = new Counter('http_status_errors');

export const options = {
  scenarios: {
    single_model: {
      executor: 'ramping-vus',
      startTime: '0s',
      stages: [
        { duration:'1m', target:5 },
        { duration:'2m', target:15 },
        { duration:'3m', target:30 },
        { duration:'3m', target:30 },
        { duration:'1m', target:0 },
      ],
      exec:'singleModel',
    },

    alternating_models: {
      executor:'ramping-vus',
      startTime:'11m',
      stages:[
        { duration:'1m', target:5 },
        { duration:'2m', target:15 },
        { duration:'3m', target:30 },
        { duration:'3m', target:30 },
        { duration:'1m', target:0 },
      ],
      exec:'alternatingModels',
    },

    batch_endpoint:{
      executor:'ramping-vus',
      startTime:'22m',
      stages:[
        { duration:'1m', target:5 },
        { duration:'2m', target:15 },
        { duration:'3m', target:30 },
        { duration:'3m', target:30 },
        { duration:'1m', target:0 },
      ],
      exec:'batchInference',
    },
  },

  thresholds:{
    http_req_failed:['rate<0.001'],
    single_inference_latency:['p(95)<4000','p(99)<5000'],
    alternating_inference_latency:['p(95)<4000','p(99)<5000'],
    batch_inference_latency:['p(95)<4000','p(99)<5000'],
  },

  summaryTrendStats:[
    'avg','min','med','max','p(90)','p(95)','p(99)'
  ],
};

const image = open('./immagine.png','b');
const batch = open('./batch.bin','b');

function checkResponse(res, metric, bodyCheck=true){
  metric.add(res.timings.duration);

  const ok = check(res,{
    'status 200': r=>r.status===200,
    'valid body': r=>{
      if(!bodyCheck) return r.body && r.body.length>0;
      try{
        return r.json().predicted_class !== undefined;
      }catch(e){
        return false;
      }
    },
  });

  failedRequests.add(!ok);

  if(res.status!==200){
    statusErrors.add(1);
    console.log(`ERROR ${res.status}: ${res.body}`);
  }
}

export function singleModel(){
  const res=http.post(
    'http://192.168.17.37:31978/predict_encoded?model=simple-cnn',
    image,
    {
      headers:{
        Host:'image-api.default.example.com',
        'Content-Type':'image/png'
      },
      timeout:'20s'
    }
  );
  checkResponse(res,singleLatency,true);
}

export function alternatingModels(){
  const model=__ITER%2===0?'simple-cnn':'simple-cnn-test';

  const res=http.post(
    `http://192.168.17.37:31978/predict_encoded?model=${model}`,
    image,
    {
      headers:{
        Host:'image-api.default.example.com',
        'Content-Type':'image/png'
      },
      timeout:'20s'
    }
  );

  checkResponse(res,alternatingLatency,true);
}

export function batchInference(){
  const res=http.post(
    'http://192.168.17.37:31978/predict_batch_encoded?models=simple-cnn,simple-cnn-test',
    batch,
    {
      headers:{
        Host:'image-api.default.example.com',
        'Content-Type':'application/octet-stream',
        'X-Image-Sizes':'674,674'
      },
      timeout:'20s'
    }
  );

  checkResponse(res,batchLatency,false);
}

export function handleSummary(data){
  return {
    "kserve-inference-report.html":htmlReport(data),
    stdout:textSummary(data,{indent:" ",enableColors:true}),
  };
}
