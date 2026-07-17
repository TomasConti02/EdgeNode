import asyncio, datetime, logging, os, pickle, sys, uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Response, File, UploadFile, Form
from sklearn.neighbors import NearestNeighbors
import redis.asyncio as redis
from datetime import timezone

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("queue-pvc")

# ---- Redis configuration ----
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_IMAGE_TTL = int(os.getenv("REDIS_IMAGE_TTL", "3600"))   # 1 hour default
STATE_KEY = "detector_state"
FALLBACK_FILE = "/data/queue/detector_state.pkl"
redis_client = None

class RealTimeOODDetector:
    def __init__(self, cent, inv_cov, win_sz, batch_sz, init_th=None, smooth=0.9, min_s=50, perc=90, max_perc=95, safe_th=None, max_drop=5, max_up=5, med_win=5, smooth_safe=False):
        self.cent = cent
        self.inv_cov = inv_cov
        self.knn = NearestNeighbors(n_neighbors=1).fit(cent)
        self.dist_buf = deque(maxlen=win_sz)
        self.batch_cnt = 0
        self.th = init_th
        self.safe_th = safe_th
        self.batch_sz = batch_sz
        self.smooth = smooth
        self.min_s = min_s
        self.perc = perc
        self.max_perc = max_perc
        self.max_drop = max_drop
        self.max_up = max_up
        self.drop_cnt = self.up_cnt = 0
        self.med_win = med_win
        self.batch_percs = deque(maxlen=med_win)
        self.smooth_safe = smooth_safe
        if inv_cov is None:
            raise ValueError("inv_cov required")
        if safe_th is None and smooth_safe:
            raise ValueError("smooth_safe needs safe_th")
        self.th_flow = deque(maxlen=max(max_drop, max_up)+1)
        self.th_trend = []
        self.fp_cor = self.fp_wrong = 0
        self.cnt = 0
        self.max_hist = 10000
        self.hist = {k: deque(maxlen=self.max_hist) for k in
                     ("time","dist","th","pred","gt","trend")}

    def _raw_th(self, dists):
        return None if len(dists)<self.min_s else np.percentile(self.dist_buf, self.perc)

    def _fallback(self, dists):
        med = np.median(dists)
        return med + 2.0 * np.median(np.abs(dists - med))

    def _update_th(self):
        if len(self.dist_buf) < self.min_s:
            return
        buf = np.asarray(self.dist_buf)
        thr = self._raw_th(buf) or self._fallback(buf)
        self.batch_percs.append(thr)
        thr_med = np.median(self.batch_percs) if len(self.batch_percs)>=2 else thr
        cand = min(thr_med, np.percentile(buf, self.max_perc))
        if self.th is None:
            self.th, trend = cand, "INIT"
        else:
            if cand > self.th:
                self.up_cnt += 1; self.drop_cnt = 0; trend = "UP"
            elif cand < self.th:
                self.drop_cnt += 1; self.up_cnt = 0; trend = "DOWN"
            else:
                self.up_cnt = self.drop_cnt = 0; trend = "SAME"
            target, reset = cand, False
            if self.safe_th is not None and (self.drop_cnt>=self.max_drop or self.up_cnt>=self.max_up):
                target, trend = self.safe_th, "RESET_SAFE"
                self.drop_cnt = self.up_cnt = 0
                if not self.smooth_safe:
                    reset = True
            elif self.drop_cnt >= self.max_drop and self.safe_th is None:
                idx = self.drop_cnt + 1
                if len(self.th_flow) >= idx:
                    target, trend = self.th_flow[-idx], "RESET_DOWN"
                else:
                    trend = "SAME"
                self.drop_cnt = self.up_cnt = 0
            elif self.up_cnt >= self.max_up and self.safe_th is None:
                idx = self.up_cnt + 1
                if len(self.th_flow) >= idx:
                    target, trend = self.th_flow[-idx], "RESET_UP"
                else:
                    trend = "SAME"
                self.up_cnt = self.drop_cnt = 0
            self.th = target if reset else self.smooth*self.th + (1-self.smooth)*target
        self.th_flow.append(cand)
        self.th_trend.append(trend)

    def process(self, emb, true_lab=None, correct=None):
        emb = np.asarray(emb).reshape(1,-1)
        idx = self.knn.kneighbors(emb, return_distance=False)[0][0]
        diff = emb.ravel() - self.cent[idx]
        dist = float(np.sqrt(diff @ self.inv_cov @ diff.T))
        self.dist_buf.append(dist)
        self.batch_cnt += 1
        self.cnt += 1
        if self.batch_cnt >= self.batch_sz:
            self.batch_cnt = 0
            self._update_th()
        if self.th is None and len(self.dist_buf)>=self.min_s:
            buf = np.asarray(self.dist_buf)
            thr = self._raw_th(buf) or self._fallback(buf)
            self.th = min(thr, np.percentile(buf, self.max_perc))
        is_ood = self.th is not None and dist > self.th
        trend = self.th_trend[-1] if self.th_trend else "INIT"
        self.hist["time"].append(self.cnt-1)
        self.hist["dist"].append(dist)
        self.hist["th"].append(self.th if self.th is not None else np.nan)
        self.hist["pred"].append(int(is_ood))
        self.hist["gt"].append(0 if true_lab is None else true_lab)
        self.hist["trend"].append(trend)
        if true_lab == 0 and is_ood:
            if correct is True or correct == 1:
                self.fp_cor += 1
            elif correct is False or correct == 0:
                self.fp_wrong += 1
        return dist, is_ood, self.th

    def get_state(self):
        return {k: getattr(self, k) for k in ( "knn","cent","inv_cov","dist_buf","batch_cnt","th","safe_th", "batch_sz","smooth","min_s","perc","max_perc","max_drop","max_up",
            "drop_cnt","up_cnt","med_win","batch_percs","smooth_safe", "th_flow","th_trend","fp_cor","fp_wrong","cnt","hist")}

    def set_state(self, st):
        for k, v in st.items():
            if k == "hist":
                new_hist = {}
                for hk, hl in v.items():
                    new_hist[hk] = deque(hl, maxlen=self.hist[hk].maxlen)
                setattr(self, k, new_hist)
            elif k in ("dist_buf", "batch_percs", "th_flow"):
                maxlen = getattr(self, k).maxlen
                setattr(self, k, deque(v, maxlen=maxlen))
            else:
                setattr(self, k, v)
#####################################################################################################################
# ---- App setup ----
detector = None
queue = asyncio.Queue(maxsize=5000)
worker = None

def init_detector():
    try:
        cent = np.load(os.getenv("CENTROIDS_PATH", "centroids.npy"))
        inv = np.load(os.getenv("INV_COV_MATRIX_PATH", "inv_cov_matrix.npy"))
    except FileNotFoundError as e:
        log.error(f"Missing file: {e}")
        raise RuntimeError("Missing centroids or inv_cov") from e
    init_th = float(os.getenv("OOD_INITIAL_THRESHOLD", "0.5"))
    params = {
        "cent": cent, "inv_cov": inv,
        "win_sz": int(os.getenv("OOD_WINDOW_SIZE", "50")),
        "batch_sz": int(os.getenv("OOD_BATCH_SIZE", "5")),
        "init_th": init_th,
        "smooth": float(os.getenv("OOD_SMOOTHING", "0.95")),
        "min_s": int(os.getenv("OOD_MIN_SAMPLES", "20")),
        "perc": float(os.getenv("OOD_PERCENTILE", "95")),
        "max_perc": float(os.getenv("OOD_MAX_PERCENTILE", "99")),
        "safe_th": float(os.getenv("OOD_SAFE_THRESHOLD", str(init_th))),
        "max_drop": int(os.getenv("OOD_MAX_CONSECUTIVE_DROPS", "3")),
        "max_up": int(os.getenv("OOD_MAX_CONSECUTIVE_UPS", "3")),
        "med_win": int(os.getenv("OOD_MEDIAN_WINDOW", "5")),
        "smooth_safe": os.getenv("OOD_SMOOTH_SAFETY_TH", "true").lower()=="true",
    }
    return RealTimeOODDetector(**params)

async def save_state():
    if detector is None:
        return
    log.info("Saving state to Redis and fallback file...")
    state = detector.get_state()
    serialized = pickle.dumps(state)
    try:
        await asyncio.wait_for(redis_client.set(STATE_KEY, serialized), timeout=2.0)
        log.info("State saved to Redis")
    except Exception as e:
        log.warning(f"Redis save failed: {e}")
    try:
        os.makedirs(os.path.dirname(FALLBACK_FILE), exist_ok=True)
        with open(FALLBACK_FILE, "wb") as f:
            pickle.dump(state, f)
        log.info(f"State saved to fallback file {FALLBACK_FILE}")
    except Exception as e:
        log.error(f"Failed to write fallback file: {e}")

async def restore_state():
    global detector
    if redis_client is not None:
        try:
            data = await redis_client.get(STATE_KEY)
            if data is not None:
                st = pickle.loads(data)
                detector = init_detector()
                detector.set_state(st)
                log.info("State restored from Redis")
                return
        except Exception as e:
            log.warning(f"Redis restore failed: {e}")

    if os.path.exists(FALLBACK_FILE):
        try:
            with open(FALLBACK_FILE, "rb") as f:
                st = pickle.load(f)
            detector = init_detector()
            detector.set_state(st)
            log.info(f"State restored from fallback file {FALLBACK_FILE}")
            return
        except Exception as e:
            log.warning(f"Fallback file restore failed: {e}")

    log.info("No state found, creating new detector")
    detector = init_detector()

async def worker_loop():
    log.info("Worker started")
    try:
        while True:
            item = await queue.get()
            insts = item["instances"]
            eid = item.get("event_id", "unknown")
            log.info(f"Processing {len(insts)} from {eid}")
            for i, inst in enumerate(insts):
                if asyncio.current_task().cancelled():
                    queue.task_done()
                    log.info(f"Cancelled, dropping {len(insts)-i}")
                    raise asyncio.CancelledError
                if isinstance(inst, dict):
                    true_lab = inst.get("true_label")
                    correct = inst.get("is_correct")
                    emb = inst.get("embedding")
                    if emb is None:
                        log.warning("Missing embedding, skip")
                        continue
                else:
                    emb, true_lab, correct = inst, None, None
                dist, ood, th = await asyncio.to_thread(detector.process, emb, true_lab, correct)
                if i % 10 == 0:
                    log.info(f"  {i+1}/{len(insts)} dist={dist:.3f} ood={ood}")
            queue.task_done()
            log.info(f"Finished {eid}")
    except asyncio.CancelledError:
        log.info("Worker cancelled, draining queue")
        while not queue.empty():
            try:
                queue.get_nowait(); queue.task_done()
            except asyncio.QueueEmpty:
                break
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, worker, redis_client
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    log.info(f"Redis client connected to {REDIS_HOST}:{REDIS_PORT}")
    await restore_state()
    worker = asyncio.create_task(worker_loop())
    yield
    log.info("Shutting down...")
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass
    await save_state()
    # Optionally force Redis to persist data to disk (RDB)
    try:
        await redis_client.save()
        log.info("Redis data saved to disk")
    except Exception as e:
        log.warning(f"Redis save failed: {e}")
    await redis_client.close()
    log.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

# ---------- Existing endpoints ----------
@app.post("/")
async def receive(req: Request):
    try:
        ev = await req.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    eid = req.headers.get("Ce-Id", "unknown")
    insts = ev.get("instances")
    if insts is None:
        raise HTTPException(400, "Missing 'instances'")
    try:
        await queue.put({"event_id": eid, "instances": insts})
        log.info(f"Enqueued {eid} ({len(insts)}) queue={queue.qsize()}")
    except asyncio.QueueFull:
        log.warning(f"Queue full, rejecting {eid}")
        raise HTTPException(503, "Overloaded")
    return Response(status_code=204)

@app.get("/health")
async def health():
    if detector is None:
        return {"status": "not_ready", "queue_size": queue.qsize(), "detector": None}
    det_state = {
        "threshold": detector.th,
        "safe_threshold": detector.safe_th,
        "counter": detector.cnt,
        "distance_buffer_len": len(detector.dist_buf),
        "batch_distances_pending": detector.batch_cnt,
        "consecutive_drops": detector.drop_cnt,
        "consecutive_ups": detector.up_cnt,
        "fp_correct": detector.fp_cor,
        "fp_wrong": detector.fp_wrong,
        "threshold_trend_last_5": detector.th_trend[-5:] if detector.th_trend else [],
        "last_distance": detector.hist["dist"][-1] if detector.hist["dist"] else None,
        "last_threshold": detector.hist["th"][-1] if detector.hist["th"] else None,
        "history_size": len(detector.hist["time"]),
    }
    return {"status": "ok", "queue_size": queue.qsize(), "detector": det_state}

# ---------- New image‑storage endpoints ----------
@app.post("/store_image")
async def store_image( file: UploadFile = File(...), ttl: Optional[int] = Form(None), metadata: Optional[str] = Form("") ): #store images in redis with a time to live 
    if redis_client is None:
        raise HTTPException(503, "Redis not available")
    img_bytes = await file.read()
    img_id = str(uuid.uuid4())
    key = f"image:{img_id}"
    ttl = ttl or REDIS_IMAGE_TTL

    await redis_client.setex(key, ttl, img_bytes)
    meta = { "filename": file.filename, "content_type": file.content_type, "timestamp": datetime.datetime.now(timezone.utc).isoformat(), "ttl": str(ttl), "metadata": metadata, }
    meta_key = f"image:{img_id}:meta"
    await redis_client.hset(meta_key, mapping=meta)
    await redis_client.expire(meta_key, ttl)
    return {"image_id": img_id, "ttl": ttl}
"""
# ---------- New image‑storage endpoints ----------
@app.post("/store_image")
async def store_image( 
    file: UploadFile = File(...), 
    ttl: Optional[int] = Form(None), 
    metadata: Optional[str] = Form(""),
    image_id: Optional[str] = Form(None)  # <--- AGGIUNTO: parametro opzionale nel Form
): #store images in redis with a time to live 
    if redis_client is None:
        raise HTTPException(503, "Redis not available")
    
    img_bytes = await file.read()
    
    # <--- MODIFICATO: Se il chiamante ha passato un ID, usa quello, altrimenti creane uno nuovo
    img_id = image_id if image_id else str(uuid.uuid4())
    
    key = f"image:{img_id}"
    ttl = ttl or REDIS_IMAGE_TTL

    await redis_client.setex(key, ttl, img_bytes)
    meta = { 
        "filename": file.filename or "unknown", 
        "content_type": file.content_type or "application/octet-stream", 
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), 
        "ttl": str(ttl), 
        "metadata": metadata or "", 
    }
    meta_key = f"image:{img_id}:meta"
    await redis_client.hset(meta_key, mapping=meta)
    await redis_client.expire(meta_key, ttl)
    
    return {"image_id": img_id, "ttl": ttl, "redis_key": key}
"""


@app.get("/image/{image_id}")
async def get_image(image_id: str):
    """Retrieve an image by its ID."""
    if redis_client is None:
        raise HTTPException(503, "Redis not available")
    key = f"image:{image_id}"
    img_data = await redis_client.get(key)
    if img_data is None:
        raise HTTPException(404, "Image not found or expired")
    meta_key = f"image:{image_id}:meta"
    meta = await redis_client.hgetall(meta_key)
    content_type = meta.get(b"content_type", b"application/octet-stream").decode()
    return Response(content=img_data, media_type=content_type)

@app.delete("/image/{image_id}")
async def delete_image(image_id: str):
    """Delete an image and its metadata from Redis."""
    if redis_client is None:
        raise HTTPException(503, "Redis not available")
    key = f"image:{image_id}"
    meta_key = f"image:{image_id}:meta"
    deleted = await redis_client.delete(key, meta_key)
    if deleted == 0:
        raise HTTPException(404, "Image not found")
    return {"status": "deleted"}
