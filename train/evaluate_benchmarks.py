"""Evaluate MonsoonResUNet on held-out validation years (RMSE, ACC, non-negativity, and monsoon shocks)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.model import MonsoonResUNet, rollout_10_days

DATA_PATH = ROOT / "data" / "india_climate_train.nc"
CKPT_PATH = ROOT / "weights" / "monsoon_unet.pt"
STATIC_PATH = ROOT / "assets" / "static_maps.npy"
TRAIN_DAYS = 5479  # 2005-01-01 .. 2019-12-31 (15 years); validation is 2020..2021 (730 days)
LEADS = (1, 4, 7, 10)


def compute_climatology(train_phys: np.ndarray, train_doy: np.ndarray) -> dict[int, np.ndarray]:
    """Compute per-day-of-year climatology [6, 32, 36] from training split only."""
    clim = {}
    for d in range(1, 367):
        mask = train_doy == d
        if np.any(mask):
            clim[d] = train_phys[mask].mean(axis=0)
        else:
            clim[d] = train_phys[train_doy == 365].mean(axis=0)
    return clim


def compute_acc(pred_anom: np.ndarray, true_anom: np.ndarray) -> np.ndarray:
    """Compute Anomaly Correlation Coefficient per channel across (N, H, W)."""
    n_ch = pred_anom.shape[1]
    acc = np.zeros(n_ch, dtype=np.float64)
    for c in range(n_ch):
        p = pred_anom[:, c].reshape(-1).astype(np.float64)
        t = true_anom[:, c].reshape(-1).astype(np.float64)
        p_c = p - p.mean()
        t_c = t - t.mean()
        denom = np.sqrt(np.sum(p_c ** 2) * np.sum(t_c ** 2))
        acc[c] = float(np.sum(p_c * t_c) / denom) if denom > 1e-12 else 0.0
    return acc


def box_mask(lats: np.ndarray, lons: np.ndarray, lat_range: tuple, lon_range: tuple) -> np.ndarray:
    """Boolean [32, 36] mask for a lat/lon box."""
    lat_m = (lats >= lat_range[0]) & (lats <= lat_range[1])
    lon_m = (lons >= lon_range[0]) & (lons <= lon_range[1])
    return np.outer(lat_m, lon_m)


def dilate_mask(mask: np.ndarray, cells: int = 3) -> np.ndarray:
    """Dilate a 2D boolean mask by `cells` grid cells (Chebyshev distance)."""
    H, W = mask.shape
    ys, xs = np.where(mask)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = np.zeros_like(mask, dtype=bool)
    for y, x in zip(ys, xs):
        out |= (np.abs(yy - y) <= cells) & (np.abs(xx - x) <= cells)
    return out


@torch.no_grad()
def main() -> None:
    device = torch.device("cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    stats = ckpt["stats"]
    channels = stats["channels"]
    static_raw = np.load(STATIC_PATH).astype(np.float32)

    model = MonsoonResUNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = xr.open_dataset(DATA_PATH)
    data_phys = ds["data"].values.astype(np.float32)
    doy_all = ds["time"].dt.dayofyear.values.astype(np.int32)
    times = ds["time"].values
    lats = ds["lat"].values.astype(np.float32)
    lons = ds["lon"].values.astype(np.float32)
    ds.close()

    train_phys, val_phys = data_phys[:TRAIN_DAYS], data_phys[TRAIN_DAYS:]
    train_doy, val_doy = doy_all[:TRAIN_DAYS], doy_all[TRAIN_DAYS:]
    val_times = times[TRAIN_DAYS:]

    clim_by_doy = compute_climatology(train_phys, train_doy)

    steps = 10
    n_win = len(val_phys) - steps
    preds_all, targets_all, pers_all, clim_all = [], [], [], []

    for start in range(0, n_win, 64):
        end = min(start + 64, n_win)
        x0 = torch.from_numpy(val_phys[start:end]).to(device)
        doy = torch.from_numpy(val_doy[start:end].astype(np.float32)).to(device)
        pred = rollout_10_days(model, x0, static_raw, stats, doy, steps=steps).cpu().numpy()

        tgt = np.stack([val_phys[i + 1 : i + 1 + steps] for i in range(start, end)], axis=0)
        pers = np.repeat(val_phys[start:end, None, ...], steps, axis=1)
        clim = np.stack(
            [
                np.stack([clim_by_doy[int(val_doy[i + 1 + k])] for k in range(steps)], axis=0)
                for i in range(start, end)
            ],
            axis=0,
        )

        preds_all.append(pred)
        targets_all.append(tgt)
        pers_all.append(pers)
        clim_all.append(clim)

    preds_arr = np.concatenate(preds_all, axis=0)
    targets_arr = np.concatenate(targets_all, axis=0)
    pers_arr = np.concatenate(pers_all, axis=0)
    clim_arr = np.concatenate(clim_all, axis=0)

    # Non-negativity verification
    tp_min = float(preds_arr[:, :, 1].min())
    tcwv_min = float(preds_arr[:, :, 5].min())
    finite_ok = bool(np.all(np.isfinite(preds_arr)))
    print(f"Validation Rollout Invariants ({n_win} windows x 10 days):")
    print(f"  All finite: {finite_ok} | min(tp)={tp_min:.6f} | min(tcwv)={tcwv_min:.6f}")
    assert finite_ok and tp_min >= 0.0 and tcwv_min >= 0.0

    # Summary Table: RMSE & ACC at Day 1, 4, 7, 10
    print("\n=== Validation Benchmark Summary (Model vs Persistence vs Climatology) ===")
    header = (
        f"{'Lead':>4s} | {'Var':>4s} | {'RMSE_Model':>10s} | {'RMSE_Pers':>10s} | "
        f"{'RMSE_Clim':>10s} | {'ACC_Model':>9s} | {'ACC_Pers':>9s}"
    )
    print(header)
    print("-" * len(header))

    day4_beats_pers = True
    for lead in LEADS:
        k = lead - 1
        rmse_m = np.sqrt(np.mean((preds_arr[:, k] - targets_arr[:, k]) ** 2, axis=(0, 2, 3)))
        rmse_p = np.sqrt(np.mean((pers_arr[:, k] - targets_arr[:, k]) ** 2, axis=(0, 2, 3)))
        rmse_c = np.sqrt(np.mean((clim_arr[:, k] - targets_arr[:, k]) ** 2, axis=(0, 2, 3)))

        acc_m = compute_acc(preds_arr[:, k] - clim_arr[:, k], targets_arr[:, k] - clim_arr[:, k])
        acc_p = compute_acc(pers_arr[:, k] - clim_arr[:, k], targets_arr[:, k] - clim_arr[:, k])

        for c_i, name in enumerate(channels):
            if lead == 4 and rmse_m[c_i] >= rmse_p[c_i]:
                day4_beats_pers = False
            print(
                f"D{lead:02d}  | {name:>4s} | {rmse_m[c_i]:10.4f} | {rmse_p[c_i]:10.4f} | "
                f"{rmse_c[c_i]:10.4f} | {acc_m[c_i]:9.4f} | {acc_p[c_i]:9.4f}"
            )
        print("-" * len(header))

    print(f"Day 4 RMSE beats Persistence on ALL 6 channels: {day4_beats_pers}")

    # Monsoon Shock Causality Tests on a real July validation state
    july_idx = int(np.where(np.datetime_as_string(val_times, unit="M") == "2020-07")[0][14])
    x0_val = val_phys[july_idx]
    doy_val = float(val_doy[july_idx])
    base_roll = rollout_10_days(model, x0_val, static_raw, stats, doy_val, steps=10).cpu().numpy()

    # Zero-shock check
    zero_roll = rollout_10_days(model, x0_val + np.zeros_like(x0_val), static_raw, stats, doy_val, steps=10).cpu().numpy()
    print(f"\nZero-Shock max |diff| on {str(val_times[july_idx])[:10]}: {np.max(np.abs(zero_roll - base_roll)):.1e}")

    bob_mask = box_mask(lats, lons, (8.0, 20.0), (82.0, 94.0))
    as_mask = box_mask(lats, lons, (8.0, 20.0), (65.0, 75.0))

    shocks = [
        ("Bay of Bengal +2.5 K (t2m)", 0, 2.5, bob_mask),
        ("Arabian Sea -3.0 m/s (u10)", 2, -3.0, as_mask),
    ]
    print("\n=== Monsoon Shock Causality Tests (2020-07-15) ===")
    for label, ch_idx, delta_val, mask in shocks:
        x0_pert = x0_val.copy()
        x0_pert[ch_idx, mask] += delta_val
        pert_roll = rollout_10_days(model, x0_pert, static_raw, stats, doy_val, steps=10).cpu().numpy()
        diff = pert_roll - base_roll

        near_mask = dilate_mask(mask, cells=3)
        d1_energy = np.sum(diff[0] ** 2, axis=0)
        d1_local_frac = float(np.sum(d1_energy[near_mask]) / max(np.sum(d1_energy), 1e-12))
        d4_tp_diff = float(np.mean(np.abs(diff[3, 1, near_mask])))
        d10_max_diff = float(np.max(np.abs(diff[9])))
        print(
            f"  {label}: D1 local energy (<=3 cells)={d1_local_frac*100:.1f}% | "
            f"D4 mean |delta tp| near box={d4_tp_diff:.4f} mm/day | D10 max |delta|={d10_max_diff:.4f}"
        )


if __name__ == "__main__":
    main()
