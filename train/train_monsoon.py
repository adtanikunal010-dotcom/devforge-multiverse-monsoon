"""Train MonsoonResUNet on data/india_climate_train.nc with 3-step rollout MSE loss."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import xarray as xr

try:
    from .model import MonsoonResUNet, normalize_static_maps, rollout_10_days
except ImportError:  # Support direct execution: python train/train_monsoon.py
    from model import MonsoonResUNet, normalize_static_maps, rollout_10_days


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "india_climate_train.nc"
STATS_PATH = ROOT / "assets" / "stats.json"
CKPT_PATH = ROOT / "weights" / "monsoon_unet.pt"

TRAIN_DAYS = 5478  # First 15 years (2005-01-01 .. 2019-12-31)
ROLLOUT_TRAIN_STEPS = 3
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4


class RolloutWindowDataset(Dataset):
    """Sliding window dataset returning normalized [6, 32, 36] start state, doy, and K targets."""

    def __init__(self, data_norm: np.ndarray, doy: np.ndarray, horizon: int = 3) -> None:
        self.data_norm = torch.from_numpy(data_norm).float()
        self.doy = torch.from_numpy(doy).float()
        self.horizon = horizon
        self.n = len(data_norm) - horizon

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x0 = self.data_norm[idx]
        targets = self.data_norm[idx + 1 : idx + 1 + self.horizon]
        return x0, targets, self.doy[idx]


def compute_rollout_loss(
    model: MonsoonResUNet,
    x0_norm: torch.Tensor,
    targets_norm: torch.Tensor,
    doy: torch.Tensor,
    static_norm: torch.Tensor,
    channel_weights: torch.Tensor,
    tp_min_norm: float,
) -> torch.Tensor:
    """Compute weighted MSE over a multi-step autoregressive rollout in normalized space."""
    B, _, H, W = x0_norm.shape
    static_b = static_norm.expand(B, 2, H, W)
    doy_b = doy.view(B, 1, 1, 1)

    curr = x0_norm
    total_loss = torch.zeros((), device=x0_norm.device, dtype=x0_norm.dtype)
    steps = targets_norm.shape[1]

    for step in range(steps):
        doy_sin = torch.sin(2.0 * math.pi * (doy_b + step + 1) / 365.25).expand(B, 1, H, W)
        inp = torch.cat([curr, static_b, doy_sin], dim=1)
        pred = model(inp)
        pred = torch.cat(
            [pred[:, :1], torch.clamp(pred[:, 1:2], min=tp_min_norm), pred[:, 2:]],
            dim=1,
        )
        sq_err = (pred - targets_norm[:, step]) ** 2
        data_loss = (sq_err * channel_weights).mean()
        grad_h = (
            (pred[:, :, 1:, :] - pred[:, :, :-1, :])
            - (targets_norm[:, step, :, 1:, :] - targets_norm[:, step, :, :-1, :])
        )
        grad_w = (
            (pred[:, :, :, 1:] - pred[:, :, :, :-1])
            - (targets_norm[:, step, :, :, 1:] - targets_norm[:, step, :, :, :-1])
        )
        spatial_loss = 0.5 * (
            (grad_h.square() * channel_weights).mean()
            + (grad_w.square() * channel_weights).mean()
        )
        total_loss = total_loss + data_loss + 0.1 * spatial_loss
        curr = pred

    return total_loss / steps


@torch.no_grad()
def evaluate_10day_rmse(
    model: MonsoonResUNet,
    val_phys: np.ndarray,
    val_doy: np.ndarray,
    static_raw: np.ndarray,
    stats: dict,
    device: torch.device,
) -> None:
    """Print validation RMSE at Day 1, Day 5, and Day 10 vs persistence baseline."""
    model.eval()
    steps = 10
    n_win = len(val_phys) - steps

    preds_all = []
    targets_all = []
    pers_all = []

    for start in range(0, n_win, 64):
        end = min(start + 64, n_win)
        x0 = torch.from_numpy(val_phys[start:end]).to(device)
        doy = torch.from_numpy(val_doy[start:end]).to(device)
        pred = rollout_10_days(model, x0, static_raw, stats, doy, steps=steps).cpu().numpy()

        tgt = np.stack([val_phys[i + 1 : i + 1 + steps] for i in range(start, end)], axis=0)
        pers = np.repeat(val_phys[start:end, None, ...], steps, axis=1)

        preds_all.append(pred)
        targets_all.append(tgt)
        pers_all.append(pers)

    preds_arr = np.concatenate(preds_all, axis=0)
    targets_arr = np.concatenate(targets_all, axis=0)
    pers_arr = np.concatenate(pers_all, axis=0)

    channels = stats["channels"]
    for lead in (1, 4, 8):
        idx = lead - 1
        model_rmse = np.sqrt(np.mean((preds_arr[:, idx] - targets_arr[:, idx]) ** 2, axis=(0, 2, 3)))
        pers_rmse = np.sqrt(np.mean((pers_arr[:, idx] - targets_arr[:, idx]) ** 2, axis=(0, 2, 3)))
        print(f"\nValidation RMSE @ Day {lead} (physical units):")
        for c_i, name in enumerate(channels):
            print(f"  {name:>4s}: model={model_rmse[c_i]:10.4f} | persistence={pers_rmse[c_i]:10.4f}")


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    ds = xr.open_dataset(DATA_PATH)
    data_phys = ds["data"].values.astype(np.float32)  # [N, 6, 32, 36]
    static_raw = ds["static"].values.astype(np.float32)  # [2, 32, 36]
    doy_all = ds["time"].dt.dayofyear.values.astype(np.float32)
    ds.close()

    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 6, 1, 1)
    data_norm = (data_phys - mean) / std

    # Chronological split: first 15 years train, remaining 2 years validation
    train_norm, val_norm = data_norm[:TRAIN_DAYS], data_norm[TRAIN_DAYS:]
    train_doy, val_doy = doy_all[:TRAIN_DAYS], doy_all[TRAIN_DAYS:]
    val_phys = data_phys[TRAIN_DAYS:]

    train_ds = RolloutWindowDataset(train_norm, train_doy, horizon=ROLLOUT_TRAIN_STEPS)
    val_ds = RolloutWindowDataset(val_norm, val_doy, horizon=ROLLOUT_TRAIN_STEPS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model_config = {"in_channels": 9, "out_channels": 6, "base_ch": 32}
    model = MonsoonResUNet(**model_config).to(device)
    static_norm = normalize_static_maps(static_raw, device=device)

    # Weight precipitation (channel index 1) by 1.5x
    channel_weights = torch.tensor(
        [1.0, 1.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device
    ).view(1, 6, 1, 1)
    tp_min_norm = -float(stats["mean"][1]) / float(stats["std"][1])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    t_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x0, tgt, doy in train_loader:
            x0 = x0.to(device)
            tgt = tgt.to(device)
            doy = doy.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = compute_rollout_loss(model, x0, tgt, doy, static_norm, channel_weights, tp_min_norm)
            loss.backward()
            optimizer.step()

            bs = x0.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x0, tgt, doy in val_loader:
                x0 = x0.to(device)
                tgt = tgt.to(device)
                doy = doy.to(device)
                loss = compute_rollout_loss(model, x0, tgt, doy, static_norm, channel_weights, tp_min_norm)
                bs = x0.size(0)
                val_loss_sum += loss.item() * bs
                val_count += bs

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "stats": stats,
                    "model_config": model_config,
                },
                CKPT_PATH,
            )

        elapsed = time.perf_counter() - t_start
        eta_s = (elapsed / epoch) * (EPOCHS - epoch)
        eta_min, eta_sec = divmod(int(round(eta_s)), 60)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | best_val={best_val_loss:.5f} | "
            f"ETA={eta_min:02d}m{eta_sec:02d}s",
            flush=True,
        )

    size_mb = CKPT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved best checkpoint to {CKPT_PATH} ({size_mb:.2f} MB)")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    evaluate_10day_rmse(model, val_phys, val_doy, static_raw, stats, device)


if __name__ == "__main__":
    main()
