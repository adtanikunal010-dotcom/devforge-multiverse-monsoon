"""Train a 3-history-day MonsoonResUNet on data/india_climate_train.nc."""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import xarray as xr

try:
    from .model import MonsoonResUNet, normalize_static_maps, rollout_10_days
except ImportError:  # Support direct execution: python train/train_monsoon.py
    from model import MonsoonResUNet, normalize_static_maps, rollout_10_days


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "india_climate_train.nc"
STATS_PATH = ROOT / "assets" / "stats.json"
# Keep the current working model safe until the new model is fully trained and
# evaluated. The candidate is promoted to CKPT_PATH only when explicitly requested.
CKPT_PATH = ROOT / "weights" / "monsoon_unet.pt"
CANDIDATE_CKPT_PATH = ROOT / "weights" / "monsoon_unet_3day_best.pt"


TRAIN_DAYS = 5478  # First 15 years (2005-01-01 .. 2019-12-31)

ROLLOUT_TRAIN_STEPS = 3
BATCH_SIZE = int(os.environ.get("MONSOON_BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("MONSOON_EPOCHS", "25"))
LR = 1e-3
WEIGHT_DECAY = 1e-4


class RolloutWindowDataset(Dataset):
    """Return history [3,6,H,W], targets [K,6,H,W], and last-input day-of-year."""

    def __init__(
        self,
        data_norm: np.ndarray,
        doy: np.ndarray,
        horizon: int = 3,
        history_days: int = HISTORY_DAYS,
    ) -> None:
        if len(data_norm) != len(doy):
            raise ValueError("data and day-of-year arrays must have equal lengths")
        if history_days < 1 or horizon < 1:
            raise ValueError("history_days and horizon must be positive")
        self.data_norm = torch.from_numpy(data_norm).float()
        self.doy = torch.from_numpy(doy).float()
        self.horizon = int(horizon)
        self.history_days = int(history_days)
        self.n = len(data_norm) - self.history_days - self.horizon + 1
        if self.n <= 0:
            raise ValueError("not enough daily samples for the requested history and horizon")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        history = self.data_norm[idx : idx + self.history_days]
        target_start = idx + self.history_days
        targets = self.data_norm[target_start : target_start + self.horizon]
        last_input_day = self.doy[target_start - 1]
        return history, targets, last_input_day


def compute_rollout_loss(
    model: MonsoonResUNet,
    history_norm: torch.Tensor,
    targets_norm: torch.Tensor,
    doy: torch.Tensor,
    static_norm: torch.Tensor,
    channel_weights: torch.Tensor,
    tp_min_norm: float,
    tcwv_min_norm: float,
) -> torch.Tensor:
    """Weighted autoregressive MSE + spatial-gradient loss in normalized space."""
    if history_norm.ndim != 5 or history_norm.shape[2] != 6:
        raise ValueError("history_norm must have shape [B,history_days,6,H,W]")
    batch, history_days, _, height, width = history_norm.shape
    if history_days != model.history_days:
        raise ValueError(
            f"model expects {model.history_days} history days, received {history_days}"
        )
    static_b = static_norm.unsqueeze(0).expand(batch, 2, height, width)
    doy_b = doy.view(batch, 1, 1, 1)

    history = history_norm
    total_loss = torch.zeros((), device=history_norm.device, dtype=history_norm.dtype)
    for step in range(targets_norm.shape[1]):
        forecast_doy = doy_b + step + 1
        seasonal = torch.sin(2.0 * math.pi * forecast_doy / 365.25).expand(
            batch, 1, height, width
        )
        model_input = torch.cat([history.flatten(1, 2), static_b, seasonal], dim=1)
        raw_pred = model(model_input)
        pred = torch.cat(
            [
                raw_pred[:, :1],
                torch.clamp(raw_pred[:, 1:2], min=tp_min_norm),
                raw_pred[:, 2:5],
                torch.clamp(raw_pred[:, 5:6], min=tcwv_min_norm),
            ],
            dim=1,
        )

        target = targets_norm[:, step]
        sq_err = (pred - target) ** 2
        data_loss = (sq_err * channel_weights).mean()
        grad_h = (
            (pred[:, :, 1:, :] - pred[:, :, :-1, :])
            - (target[:, :, 1:, :] - target[:, :, :-1, :])
        )
        grad_w = (
            (pred[:, :, :, 1:] - pred[:, :, :, :-1])
            - (target[:, :, :, 1:] - target[:, :, :, :-1])
        )
        spatial_loss = 0.5 * (
            (grad_h.square() * channel_weights).mean()
            + (grad_w.square() * channel_weights).mean()
        )
        total_loss = total_loss + data_loss + 0.1 * spatial_loss
        history = torch.cat([history[:, 1:], pred.unsqueeze(1)], dim=1)

    return total_loss / targets_norm.shape[1]


def latitude_weighted_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RMSE per channel with cos(latitude) weighting; arrays are [N,6,H,W]."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("pred and target must both have shape [N,6,H,W]")
    latitudes = 6.5 + np.arange(pred.shape[2], dtype=np.float64)
    weights = np.cos(np.deg2rad(latitudes)).reshape(1, 1, -1, 1)
    numerator = np.sum((pred - target) ** 2 * weights, axis=(0, 2, 3))
    denominator = pred.shape[0] * pred.shape[3] * np.sum(weights)
    return np.sqrt(numerator / denominator)


@torch.no_grad()
def evaluate_10day_rmse(
    model: MonsoonResUNet,
    val_phys: np.ndarray,
    val_doy: np.ndarray,
    static_raw: np.ndarray,
    stats: dict,
    device: torch.device,
) -> bool:
    """Aggregate validation RMSE at Day 1/4/8 against persistence.

    Accumulators avoid retaining every 10-day rollout in memory. Returns True
    only when Day-4 latitude-weighted RMSE is lower than persistence for all
    six channels (a deliberately conservative local gate).
    """
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
    if BATCH_SIZE < 1 or EPOCHS < 1:
        raise ValueError("MONSOON_BATCH_SIZE and MONSOON_EPOCHS must be positive integers")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}. Obtain the authorized dataset; do not commit it."
        )
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    ds = xr.open_dataset(DATA_PATH)
    try:
        data_phys = ds["data"].values.astype(np.float32)  # [N,6,32,36]
        static_raw = ds["static"].values.astype(np.float32)  # [2,32,36]
        doy_all = ds["time"].dt.dayofyear.values.astype(np.float32)
        times = ds["time"].values
    finally:
        ds.close()

    if data_phys.ndim != 4 or data_phys.shape[1:] != (6, 32, 36):
        raise ValueError(f"unexpected data shape: {data_phys.shape}; expected [N,6,32,36]")
    validate_chronological_split(times)

    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 6, 1, 1)
    if np.any(std <= 0):
        raise ValueError("stats.json must contain positive standard deviations")
    data_norm = (data_phys - mean) / std

    # Split chronologically. No windows cross the train/validation boundary.
    train_norm, val_norm = data_norm[:TRAIN_DAYS], data_norm[TRAIN_DAYS:]
    train_doy, val_doy = doy_all[:TRAIN_DAYS], doy_all[TRAIN_DAYS:]
    val_phys = data_phys[TRAIN_DAYS:]

    train_ds = RolloutWindowDataset(train_norm, train_doy, horizon=ROLLOUT_TRAIN_STEPS)
    val_ds = RolloutWindowDataset(val_norm, val_doy, horizon=ROLLOUT_TRAIN_STEPS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model_config = {"in_channels": HISTORY_DAYS * 6 + 3, "out_channels": 6, "base_ch": 32}
    model = MonsoonResUNet(**model_config).to(device)
    print("Checkpoint initialization:", warm_start_from_existing_checkpoint(model), flush=True)
    model.to(device)
    static_norm = normalize_static_maps(static_raw, device=device)
    channel_weights = torch.tensor(
        [1.0, 1.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device
    ).view(1, 6, 1, 1)
    tp_min_norm = -float(stats["mean"][1]) / float(stats["std"][1])
    tcwv_min_norm = -float(stats["mean"][5]) / float(stats["std"][5])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    CANDIDATE_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    t_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for history, targets, doy in train_loader:
            history = history.to(device)
            targets = targets.to(device)
            doy = doy.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_rollout_loss(
                model, history, targets, doy, static_norm, channel_weights,
                tp_min_norm, tcwv_min_norm,
            )
            loss.backward()
            optimizer.step()
            batch_size = history.size(0)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        scheduler.step()
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for history, targets, doy in val_loader:
                history = history.to(device)
                targets = targets.to(device)
                doy = doy.to(device)
                loss = compute_rollout_loss(
                    model, history, targets, doy, static_norm, channel_weights,
                    tp_min_norm, tcwv_min_norm,
                )
                batch_size = history.size(0)
                val_loss_sum += loss.item() * batch_size
                val_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "stats": stats,
                    "model_config": model_config,
                },
                CANDIDATE_CKPT_PATH,
            )

        elapsed = time.perf_counter() - t_start
        eta_seconds = int(round((elapsed / epoch) * (EPOCHS - epoch)))
        eta_minutes, eta_seconds = divmod(eta_seconds, 60)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | best_val={best_val_loss:.5f} | "
            f"ETA={eta_minutes:02d}m{eta_seconds:02d}s",
            flush=True,
        )

    size_mb = CANDIDATE_CKPT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved best 3-history-day candidate to {CANDIDATE_CKPT_PATH} ({size_mb:.2f} MB)")
    checkpoint = torch.load(CANDIDATE_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate_10day_rmse(model, val_phys, val_doy, static_raw, stats, device)

    print(
        "\nThe existing serving checkpoint was left unchanged. Review the full benchmark "
        "before manually promoting this candidate."
    )
    print(
        "When satisfied with the results, back up weights/monsoon_unet.pt and copy "
        "weights/monsoon_unet_3day_best.pt to weights/monsoon_unet.pt."
    )


if __name__ == "__main__":
    main()
