"""Round 1 inference server backed by trained MonsoonResUNet (offline CPU)."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from train.model import MonsoonResUNet, rollout_10_days

torch.set_num_threads(min(4, torch.get_num_threads()))

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "monsoon_unet.pt"
STATS_PATH = ROOT / "assets" / "stats.json"
STATIC_PATH = ROOT / "assets" / "static_maps.npy"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_NAME = "monsoon-resunet"

with open(STATS_PATH, "r", encoding="utf-8") as _f:
    _FILE_STATS = json.load(_f)

# Prefer the exact statistics stored with the checkpoint. This prevents a newly
# prepared ERA5 stats.json from silently changing the normalization of an older
# checkpoint if the candidate model has not yet been promoted.
_CKPT = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
_STATS = _CKPT.get("stats", _FILE_STATS)
_STATIC_MAPS = np.load(STATIC_PATH).astype("f4")
_MODEL_CFG = _CKPT.get("model_config", {"in_channels": 9, "out_channels": 6, "base_ch": 32})
_MODEL = MonsoonResUNet(**_MODEL_CFG)
_MODEL.load_state_dict(_CKPT["model_state_dict"])
_MODEL.to("cpu")
_MODEL.eval()


def _arr(x):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        import base64
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


def _doy_from_sincos(doy_sincos) -> float:
    """Recover day-of-year in [0, 365) from [sin(2*pi*doy/365), cos(2*pi*doy/365)]."""
    sin_v, cos_v = float(doy_sincos[0]), float(doy_sincos[1])
    angle = math.atan2(sin_v, cos_v)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return (angle / (2.0 * math.pi)) * 365.0


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """Forecast from [3,6,32,36] physical states, oldest first.

    A legacy 9-channel checkpoint uses the latest state; a newly trained
    21-channel checkpoint consumes all three states. The perturbation is a
    physical-unit delta applied only to the latest input state.
    """
    history = init_states.astype("f4", copy=True)
    if history.shape != (3, 6, 32, 36):
        raise ValueError(f"init_states must have shape [3,6,32,36], received {history.shape}")
    if perturbation is not None and np.any(perturbation != 0.0):
        history[-1] = history[-1] + perturbation.astype("f4", copy=False)

    if not np.all(np.isfinite(history)):
        history = np.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)

    doy = _doy_from_sincos(doy_sincos)
    with torch.no_grad():
        out_t = rollout_10_days(_MODEL, history, _STATIC_MAPS, _STATS, doy, steps=steps)

    out = out_t.cpu().numpy().astype("f4", copy=False)
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    out[:, 5] = np.maximum(out[:, 5], 0.0)
    return out


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    import time
    t0 = time.perf_counter(); body = await request.json()
    init = _arr(body["init_states"]); pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
    out = simulate(init, body["doy_sincos"], int(body["steps"]), pert)
    return {"rollout": np.asarray(out, dtype="f4").tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
