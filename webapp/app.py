"""
FastAPI wrapper around nsd_inference_attention_pcr.py. Serves a single
mobile-friendly page (static/index.html) and one JSON endpoint that runs
the 5-fold ensemble on an uploaded/recorded video and returns the
ensemble tread-depth prediction.
"""

import os
import sys
import tempfile
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEBAPP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import nsd_inference_attention_pcr as infer

infer.CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoint_1600_data_attention_excluded_2to9")
infer.FOLD_STATS_PATH = os.path.join(infer.CHECKPOINT_DIR, "fold_stats.json")
infer.CACHE_DIR = os.path.join(tempfile.gettempdir(), "nsd_inference_cache")

MAX_UPLOAD_BYTES = 150 * 1024 * 1024  # 150MB, generous for a short phone/camera clip

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    fold_checkpoint_paths, fold_stats, n_hand_features = infer._discover_checkpoints()
    device = infer.select_device()
    models = infer._get_models(fold_checkpoint_paths, n_hand_features, device)
    _state["fold_checkpoint_paths"] = fold_checkpoint_paths
    _state["fold_stats"] = fold_stats
    _state["n_hand_features"] = n_hand_features
    _state["device"] = device
    print(f"Loaded {len(models)} fold model(s) on {device}.")
    yield


app = FastAPI(title="NSD Tread Depth Estimator", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(os.path.join(WEBAPP_DIR, "static", "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(_state.get("device"))}


@app.post("/api/predict")
async def predict(video: UploadFile = File(...)):
    if not _state:
        raise HTTPException(status_code=503, detail="Model is still loading, try again in a moment.")

    suffix = os.path.splitext(video.filename or "")[1] or ".webm"
    tmp_path = None
    try:
        data = await video.read()
        if not data:
            raise HTTPException(status_code=400, detail="Received an empty file.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Video is too large (max 150MB).")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        result = infer.predict_video(
            tmp_path,
            fold_checkpoint_paths=_state["fold_checkpoint_paths"],
            fold_stats=_state["fold_stats"],
            n_hand_features=_state["n_hand_features"],
            device=_state["device"],
            use_cache=False,
        )
        return JSONResponse({
            "ensemble_pred_mm": round(result["ensemble_pred_mm"], 2),
        })
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not analyze this video: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


app.mount("/static", StaticFiles(directory=os.path.join(WEBAPP_DIR, "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5050)
