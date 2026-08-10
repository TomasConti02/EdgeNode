/*
curl -X POST "http://192.168.17.37:31978/v1/models/simple-cnn:predict" \
  -H "Host: simple-cnn-predictor.default.example.com" \
  -H "Content-Type: application/json" \
  -d "{\"raw_input_contents\": [\"$(base64 -w 0 ./immagine.png)\"]}"
*/

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';
import encoding from 'k6/encoding';
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

const rawImage = open('./immagine.png', 'b');
const base64Image = encoding.b64encode(rawImage);

const INGRESS_HOST = '192.168.17.37';
const INGRESS_PORT = '31978';

export function testModel(modelName) {
  const url = `http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${modelName}:predict`;
  const hostName = `${modelName}-predictor.default.example.com`;

  const payload = JSON.stringify({
    raw_input_contents: [base64Image]
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Host': hostName,
    },
    timeout: '15s',
  };

  const res = http.post(url, payload, params);
  
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



/*
kubectl logs simple-cnn-predictor-00001-deployment-6b97d4bcb5-6fvjm
2026-08-10 12:19:24.625 1 kserve INFO [model_server.py:register_model():406] Registering model: simple-cnn
2026-08-10 12:19:24.626 1 kserve INFO [model_server.py:setup_event_loop():286] Setting max asyncio worker threads as 32
2026-08-10 12:19:24.628 1 kserve INFO [server.py:start():161] Starting uvicorn with 8 workers
2026-08-10 12:19:24.628 1 uvicorn.error INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
2026-08-10 12:19:24.629 1 kserve INFO [server.py:start():165] Started parent process [1]
2026-08-10 12:19:24.650 1 kserve INFO [server.py:init_processes():158] Started child process [71]
2026-08-10 12:19:24.659 1 kserve INFO [server.py:init_processes():158] Started child process [72]
2026-08-10 12:19:24.666 1 kserve INFO [server.py:init_processes():158] Started child process [73]
2026-08-10 12:19:24.675 1 kserve INFO [server.py:init_processes():158] Started child process [74]
2026-08-10 12:19:24.684 1 kserve INFO [server.py:init_processes():158] Started child process [75]
2026-08-10 12:19:24.692 1 kserve INFO [server.py:init_processes():158] Started child process [76]
2026-08-10 12:19:24.700 1 kserve INFO [server.py:init_processes():158] Started child process [77]
2026-08-10 12:19:24.709 1 kserve INFO [server.py:init_processes():158] Started child process [78]
2026-08-10 12:19:24.714 1 kserve INFO [server.py:start():70] Starting gRPC server with 4 workers
2026-08-10 12:19:24.714 1 kserve INFO [server.py:start():71] Starting gRPC server on [::]:8081
2026-08-10 12:19:30.357 1 kserve INFO [server.py:keep_subprocess_alive():185] Child process [71] died
I0810 12:19:30.359414       1 fork_posix.cc:71] Other threads are currently calling into gRPC, skipping fork() handlers
2026-08-10 12:19:30.371 1 kserve INFO [server.py:keep_subprocess_alive():192] Started new child process [594]
2026-08-10 12:19:32.955 77 kserve INFO [server.py:_register_endpoints():111] OpenAI endpoints not registered
2026-08-10 12:19:32.956 77 kserve INFO [server.py:_register_endpoints():119] Time series endpoints not registered
2026-08-10 12:19:33.169 75 kserve INFO [server.py:_register_endpoints():111] OpenAI endpoints not registered
2026-08-10 12:19:33.169 75 kserve INFO [server.py:_register_endpoints():119] Time series endpoints not registered
2026-08-10 12:19:33.273 77 uvicorn.error INFO:     Started server process [77]
2026-08-10 12:19:33.274 77 uvicorn.error INFO:     Waiting for application startup.
2026-08-10 12:19:33.274 77 uvicorn.error INFO:     Application startup complete.
2026-08-10 12:19:33.360 76 kserve INFO [server.py:_register_endpoints():111] OpenAI endpoints not registered
2026-08-10 12:19:33.360 76 kserve INFO [server.py:_register_endpoints():119] Time series endpoints not registered
2026-08-10 12:19:33.395 72 kserve INFO [server.py:_register_endpoints():111] OpenAI endpoints not registered
2026-08-10 12:19:33.395 72 kserve INFO [server.py:_register_endpoints():119] Time series endpoints not registered
2026-08-10 12:19:33.455 74 kserve INFO [server.py:_register_endpoints():111] OpenAI endpoints not registered
2026-08-10 12:19:33.455 74 kserve INFO [server.py:_register_endpoints():119] Time series endpoints not registered
2026-08-10 12:19:33.482 73 kserve INFO [server.py:_register_endpoints():111] OpenAI endpoints not registered
2026-08-10 12:19:33.482 73 kserve INFO [server.py:_register_endpoints():119] Time series endpoints not registered
2026-08-10 12:19:33.490 75 uvicorn.error INFO:     Started server process [75]
2026-08-10 12:19:33.490 75 uvicorn.error INFO:     Waiting for application startup.
2026-08-10 12:19:33.491 75 uvicorn.error INFO:     Application startup complete.
2026-08-10 12:19:33.551 78 kserve INFO [server.py:_register_endpoints():111] OpenAI endpoints not registered
2026-08-10 12:19:33.551 78 kserve INFO [server.py:_register_endpoints():119] Time series endpoints not registered
2026-08-10 12:19:33.656 76 uvicorn.error INFO:     Started server process [76]
2026-08-10 12:19:33.656 76 uvicorn.error INFO:     Waiting for application startup.
2026-08-10 12:19:33.657 76 uvicorn.error INFO:     Application startup complete.
2026-08-10 12:19:33.690 72 uvicorn.error INFO:     Started server process [72]
2026-08-10 12:19:33.690 72 uvicorn.error INFO:     Waiting for application startup.
2026-08-10 12:19:33.691 72 uvicorn.error INFO:     Application startup complete.
2026-08-10 12:19:33.713 74 uvicorn.error INFO:     Started server process [74]
2026-08-10 12:19:33.713 74 uvicorn.error INFO:     Waiting for application startup.
2026-08-10 12:19:33.713 74 uvicorn.error INFO:     Application startup complete.
2026-08-10 12:19:33.728 73 uvicorn.error INFO:     Started server process [73]
2026-08-10 12:19:33.728 73 uvicorn.error INFO:     Waiting for application startup.
2026-08-10 12:19:33.728 73 uvicorn.error INFO:     Application startup complete.
2026-08-10 12:19:33.748 78 uvicorn.error INFO:     Started server process [78]
2026-08-10 12:19:33.749 78 uvicorn.error INFO:     Waiting for application startup.
2026-08-10 12:19:33.749 78 uvicorn.error INFO:     Application startup complete.
2026-08-10 12:19:35.303 594 kserve INFO [server.py:_register_endpoints():111] OpenAI endpoints not registered
2026-08-10 12:19:35.303 594 kserve INFO [server.py:_register_endpoints():119] Time series endpoints not registered
2026-08-10 12:19:35.470 594 uvicorn.error INFO:     Started server process [594]
2026-08-10 12:19:35.470 594 uvicorn.error INFO:     Waiting for application startup.
2026-08-10 12:19:35.471 594 uvicorn.error INFO:     Application startup complete.
ERROR:__mp_main__:DEBUG -  {
  key: "probabilities"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 10
      }
    }
    float_val: 0.00319806021
    float_val: 1.07854987e-06
    float_val: 0.648779
    float_val: 0.0268888697
    float_val: 0.00986028649
    float_val: 0.000152019711
    float_val: 0.00571593409
    float_val: 0.00132279412
    float_val: 0.133430809
    float_val: 0.170651153
  }
}
outputs {
  key: "predicted_class"
  value {
    dtype: DT_INT32
    tensor_shape {
      dim {
        size: 1
      }
    }
    int_val: 2
  }
}
outputs {
  key: "embedding"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 512
      }
    }
    float_val: 0
    float_val: 0.134895
    float_val: 7.39281034
    float_val: 0.0490613244
    float_val: 0
2026-08-10 12:21:54.771 uvicorn.access INFO:     10.42.0.251:0 74 - "POST /v1/models/simple-cnn%3Apredict HTTP/1.1" 200 OK
    float_val: 0.638588607
    float_val: 1.05838692
    float_val: 5.32261801
    float_val: 0.41967684
    float    float_val: 4.67811537
    float_val: 0.0371936187
  }
}
model_spec {
  name: "simple-cnn"
  version {
    value: 1
  }
  signature_name: "serving_default"
}

2026-08-10 12:21:54.771 74 kserve.trace requestId: 1cddf90f-cd85-4a20-8ab7-f5a35b36abfe, preprocess_ms: 1.295804977, explain_ms: 0, predict_ms: 844.575643539, postprocess_ms: 0.039815903
2026-08-10 12:21:54.771 74 kserve.trace kserve.io.kserve.protocol.rest.v1_endpoints.predict: 0.8487443923950195 ['http_status:200', 'http_method:POST', 'time:wall']
2026-08-10 12:21:54.771 74 kserve.trace kserve.io.kserve.protocol.rest.v1_endpoints.predict: 0.02335300000000018 ['http_status:200', 'http_method:POST', 'time:cpu']

*/
