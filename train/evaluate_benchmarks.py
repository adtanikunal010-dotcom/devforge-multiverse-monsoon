"""Evaluate MonsoonResUNet on held-out validation years (2020-2021) using R1_CONTRACT.md metrics."""

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
OUT_JSON_PATH = ROOT / "assets" / "benchmark_results.json"

TRAIN_DAYS = 5478  # First 15 years (2005-01-01 .. 2019-12-31); val starts 2020-01-01
LEADS = (1, 4, 8)
HEAVY_RAIN_THRESHOLD = 10.0  # mm/day


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


def lat_weighted_rmse(pred: np.ndarray, target: np.ndarray, lat_w: np.ndarray) -> np.ndarray:
    """Compute cos(lat)-weighted RMSE per channel across (N, H, W). lat_w has shape [1, 1, H, 1]."""
    sq_err = (pred.astype(np.float64) - target.astype(np.float64)) ** 2
    return np.sqrt(np.mean(sq_err * lat_w, axis=(0, 2, 3)))


def compute_csi(pred_tp: np.ndarray, true_tp: np.ndarray, threshold: float = HEAVY_RAIN_THRESHOLD) -> float:
    """Critical Success Index for tp >= threshold (mm/day)."""
    p_hit = pred_tp >= threshold
    t_hit = true_tp >= threshold
    hits = int(np.sum(p_hit & t_hit))
    misses = int(np.sum((~p_hit) & t_hit))
    false_alarms = int(np.sum(p_hit & (~t_hit)))
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else 0.0


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


def distance_from_mask(mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance in grid cells from True entries of `mask`."""
    H, W = mask.shape
    ys, xs = np.where(mask)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dist = np.full((H, W), np.inf, dtype=np.float64)
    for y, x in zip(ys, xs):
        d = np.maximum(np.abs(yy - y), np.abs(xx - x))
        dist = np.minimum(dist, d)
    return dist


@torch.no_grad()
def main() -> None:
    device = torch.device("cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    stats = ckpt["stats"]
    channels = stats["channels"]
    channel_std = np.asarray(stats["std"], dtype=np.float64).reshape(1, 6, 1, 1)
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

    # Normalized cos(lat) weights for [N, C, H, W]
    cos_lat = np.cos(np.deg2rad(lats)).astype(np.float64)
    lat_w = (cos_lat / cos_lat.mean()).reshape(1, 1, len(lats), 1)

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

    tp_min = float(preds_arr[:, :, 1].min())
    tcwv_min = float(preds_arr[:, :, 5].min())
    finite_ok = bool(np.all(np.isfinite(preds_arr)))
    print(f"Validation Split: train_days={TRAIN_DAYS} (2005..2019), val_start={str(val_times[0])[:10]}, windows={n_win}")
    print(f"Invariants: all_finite={finite_ok} | min(tp)={tp_min:.6f} | min(tcwv)={tcwv_min:.6f}")
    assert finite_ok and tp_min >= 0.0 and tcwv_min >= 0.0

    results = {
        "split": {
            "train_days": TRAIN_DAYS,
            "val_days": int(len(val_phys)),
            "val_start_date": str(val_times[0])[:10],
            "val_end_date": str(val_times[-1])[:10],
            "n_windows": int(n_win),
        },
        "invariants": {"all_finite": finite_ok, "min_tp": tp_min, "min_tcwv": tcwv_min},
        "leads": {},
        "shocks": {},
    }

    print("\n=== Official Latitude-Weighted RMSE & Skill vs Best Baseline (Leads 1, 4, 8) ===")
    header = (
        f"{'Lead':>4s} | {'Var':>4s} | {'RMSE_Model':>10s} | {'RMSE_Pers':>10s} | "
        f"{'RMSE_Clim':>10s} | {'Best_Base':>10s} | {'Skill_vs_Best':>13s} | {'ACC_Model':>9s}"
    )
    print(header)
    print("-" * len(header))

    day4_beats_pers = True
    for lead in LEADS:
        k = lead - 1
        rmse_m = lat_weighted_rmse(preds_arr[:, k], targets_arr[:, k], lat_w)
        rmse_p = lat_weighted_rmse(pers_arr[:, k], targets_arr[:, k], lat_w)
        rmse_c = lat_weighted_rmse(clim_arr[:, k], targets_arr[:, k], lat_w)
        acc_m = compute_acc(preds_arr[:, k] - clim_arr[:, k], targets_arr[:, k] - clim_arr[:, k])

        csi_m = compute_csi(preds_arr[:, k, 1], targets_arr[:, k, 1])
        csi_p = compute_csi(pers_arr[:, k, 1], targets_arr[:, k, 1])
        csi_c = compute_csi(clim_arr[:, k, 1], targets_arr[:, k, 1])

        lead_dict = {
            "csi_tp_10mm": {
                "model": round(csi_m, 4),
                "persistence": round(csi_p, 4),
                "climatology": round(csi_c, 4),
            },
            "variables": {},
        }

        for c_i, name in enumerate(channels):
            best_b = float(min(rmse_p[c_i], rmse_c[c_i]))
            skill = 1.0 - float(rmse_m[c_i]) / best_b
            if lead == 4 and rmse_m[c_i] >= rmse_p[c_i]:
                day4_beats_pers = False
            print(
                f"D{lead:02d}  | {name:>4s} | {rmse_m[c_i]:10.4f} | {rmse_p[c_i]:10.4f} | "
                f"{rmse_c[c_i]:10.4f} | {best_b:10.4f} | {skill * 100:+12.2f}% | {acc_m[c_i]:9.4f}"
            )
            lead_dict["variables"][name] = {
                "rmse_model": round(float(rmse_m[c_i]), 4),
                "rmse_persistence": round(float(rmse_p[c_i]), 4),
                "rmse_climatology": round(float(rmse_c[c_i]), 4),
                "rmse_best_baseline": round(best_b, 4),
                "skill_vs_best_pct": round(skill * 100.0, 2),
                "acc_model": round(float(acc_m[c_i]), 4),
            }
        print("-" * len(header))
        results["leads"][f"day_{lead}"] = lead_dict

    print("\n=== Heavy-Rain CSI (>= 10.0 mm/day on tp) ===")
    for lead in LEADS:
        c = results["leads"][f"day_{lead}"]["csi_tp_10mm"]
        print(
            f"  Day {lead}: Model={c['model']:.4f} | Persistence={c['persistence']:.4f} | "
            f"Climatology={c['climatology']:.4f}"
        )
    print(f"Day 4 RMSE beats Persistence on ALL 6 channels: {day4_beats_pers}")
    results["day4_beats_persistence_all_6"] = day4_beats_pers

    # Official Contract Shocks on a real July validation state (2020-07-15)
    july_idx = int(np.where(np.datetime_as_string(val_times, unit="M") == "2020-07")[0][14])
    x0_val = val_phys[july_idx]
    doy_val = float(val_doy[july_idx])
    base_roll = rollout_10_days(model, x0_val, static_raw, stats, doy_val, steps=10).cpu().numpy()

    zero_roll = rollout_10_days(model, x0_val + np.zeros_like(x0_val), static_raw, stats, doy_val, steps=10).cpu().numpy()
    zero_diff = float(np.max(np.abs(zero_roll - base_roll)))
    results["shocks"]["zero_shock_max_diff"] = zero_diff

    bob_mask = box_mask(lats, lons, (8.0, 20.0), (82.0, 94.0))
    as_mask = box_mask(lats, lons, (8.0, 20.0), (65.0, 75.0))

    warm_ocean_pert = np.zeros_like(x0_val)
    warm_ocean_pert[0, bob_mask] = 3.0
    warm_ocean_pert[5, bob_mask] = 0.15 * x0_val[5, bob_mask]

    monsoon_stall_pert = np.zeros_like(x0_val)
    monsoon_stall_pert[2, as_mask] = -0.40 * x0_val[2, as_mask]
    monsoon_stall_pert[3, as_mask] = -0.40 * x0_val[3, as_mask]

    print(f"\n=== Official Contract Shock Causality Tests ({str(val_times[july_idx])[:10]}) ===")
    print(f"  null (zero shock) max |diff|: {zero_diff:.1e}")
    for shock_name, pert, mask in (
        ("warm_ocean", warm_ocean_pert, bob_mask),
        ("monsoon_stall", monsoon_stall_pert, as_mask),
    ):
        pert_roll = rollout_10_days(model, x0_val + pert, static_raw, stats, doy_val, steps=10).cpu().numpy()
        resp_std = (pert_roll.astype(np.float64) - base_roll.astype(np.float64)) / channel_std
        max_std = float(np.max(np.abs(resp_std)))
        energy_by_day = np.square(resp_std).sum(axis=1)
        dist = distance_from_mask(mask)
        d1_total = float(energy_by_day[0].sum())
        d1_local = float(energy_by_day[0][dist <= 3.0].sum() / d1_total) if d1_total > 1e-12 else 0.0
        d4_tp_near = float(np.mean(np.abs((pert_roll - base_roll)[3, 1, dist <= 3.0])))
        d8_tp_near = float(np.mean(np.abs((pert_roll - base_roll)[7, 1, dist <= 3.0])))

        print(
            f"  {shock_name:>13s}: D1 local energy (<=3 cells)={d1_local*100:.1f}% | "
            f"max_std={max_std:.3f} (<=5.0) | D4 |d_tp|={d4_tp_near:.4f} | D8 |d_tp|={d8_tp_near:.4f}"
        )
        results["shocks"][shock_name] = {
            "d1_local_energy_pct": round(d1_local * 100.0, 2),
            "max_response_std": round(max_std, 4),
            "d4_mean_abs_delta_tp_mm_day": round(d4_tp_near, 4),
            "d8_mean_abs_delta_tp_mm_day": round(d8_tp_near, 4),
        }

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved structured benchmark summary to {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
