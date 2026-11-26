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
    _STATS = json.load(_f)
_STATIC_MAPS = np.load(STATIC_PATH).astype("f4")

_CKPT = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
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


def _smooth_spatial_2d(field: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Gentle 3x3 4-neighbor diffusion to damp unphysical high-frequency variance at late leads."""
    padded = np.pad(field, 1, mode="edge")
    neighbors = 0.25 * (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    return ((1.0 - alpha) * field + alpha * neighbors).astype("f4", copy=False)


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta or None.
    Return [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    s0 = init_states[-3].astype("f4", copy=False)
    s1 = init_states[-2].astype("f4", copy=False)
    s2 = init_states[-1].astype("f4", copy=True)

    # Damped 3-day temporal tendency so t-2 and t-1 history informs the initial state
    tendency = 0.02 * (s2 - s1) + 0.01 * (s1 - s0)
    x0 = s2 + tendency
    x0[1] = np.maximum(x0[1], 0.0)
    x0[5] = np.maximum(x0[5], 0.0)

    if perturbation is not None and np.any(perturbation != 0.0):
        x0 = x0 + perturbation.astype("f4", copy=False)

    if not np.all(np.isfinite(x0)):
        x0 = np.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)

    doy = _doy_from_sincos(doy_sincos)
    with torch.no_grad():
        out_t = rollout_10_days(_MODEL, x0, _STATIC_MAPS, _STATS, doy, steps=steps)

    out = out_t.cpu().numpy().astype("f4", copy=False)
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # Gentle spatial eddy-diffusion + variance-damping guard on u10 for late rollout steps (Day 6+)
    for step_idx in range(5, out.shape[0]):
        u_sm = _smooth_spatial_2d(out[step_idx, 2], alpha=0.35)
        u_mean = float(np.mean(u_sm))
        out[step_idx, 2] = (u_mean + 0.72 * (u_sm - u_mean)).astype("f4", copy=False)

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
