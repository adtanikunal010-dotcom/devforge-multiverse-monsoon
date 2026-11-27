"""Contract-aligned evaluation: latitude-weighted RMSE skill, heavy-rain CSI and shocks."""

from __future__ import annotations

import argparse
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
CANDIDATE_CKPT_PATH = ROOT / "weights" / "monsoon_unet_3day_best.pt"
STATIC_PATH = ROOT / "assets" / "static_maps.npy"
# 2005-01-01 .. 2019-12-31 inclusive.
TRAIN_DAYS = 5478
HISTORY_DAYS = 3
STEPS = 10
LEADS = (1, 4, 8)
RAIN_THRESHOLD_MM_DAY = 10.0


def compute_climatology(train_phys: np.ndarray, train_doy: np.ndarray) -> dict[int, np.ndarray]:
    """Compute training-only per-day-of-year climatology [6,H,W]."""
    clim: dict[int, np.ndarray] = {}
    fallback = train_phys[train_doy == 365]
    if not len(fallback):
        fallback = train_phys
    fallback_mean = fallback.mean(axis=0)
    for day in range(1, 367):
        selected = train_phys[train_doy == day]
        clim[day] = selected.mean(axis=0) if len(selected) else fallback_mean
    return clim


def latitude_weighted_rmse(pred: np.ndarray, target: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    """Per-channel RMSE weighted by cos(latitude), inputs [N,6,H,W]."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("pred and target must have matching shape [N,6,H,W]")
    if len(latitudes) != pred.shape[2]:
        raise ValueError("latitude coordinate length does not match grid height")
    weights = np.cos(np.deg2rad(latitudes.astype(np.float64))).reshape(1, 1, -1, 1)
    numerator = np.sum((pred.astype(np.float64) - target.astype(np.float64)) ** 2 * weights, axis=(0, 2, 3))
    denominator = pred.shape[0] * pred.shape[3] * np.sum(weights)
    return np.sqrt(numerator / denominator)


def heavy_rain_csi(pred_tp: np.ndarray, true_tp: np.ndarray, threshold: float = RAIN_THRESHOLD_MM_DAY) -> float:
    """Critical Success Index for daily precipitation >= threshold (mm/day)."""
    predicted_event = pred_tp >= threshold
    observed_event = true_tp >= threshold
    hits = int(np.count_nonzero(predicted_event & observed_event))
    false_alarms = int(np.count_nonzero(predicted_event & ~observed_event))
    misses = int(np.count_nonzero(~predicted_event & observed_event))
    denominator = hits + false_alarms + misses
    return float(hits / denominator) if denominator else float("nan")


def box_mask(lats: np.ndarray, lons: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> np.ndarray:
    lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
    lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])
    return np.outer(lat_mask, lon_mask)


def dilate_mask(mask: np.ndarray, cells: int) -> np.ndarray:
    height, width = mask.shape
    yy, xx = np.ogrid[:height, :width]
    result = np.zeros_like(mask, dtype=bool)
    for y, x in zip(*np.where(mask)):
        result |= (np.abs(yy - y) <= cells) & (np.abs(xx - x) <= cells)
    return result


def energy_radius(diff_day: np.ndarray, mask: np.ndarray, fraction: float = 0.95) -> float:
    """Smallest Chebyshev dilation radius containing the requested response energy."""
    energy = np.sum(np.asarray(diff_day, dtype=np.float64) ** 2, axis=0)
    total = float(energy.sum())
    if total <= 1e-18:
        return 0.0
    max_radius = max(mask.shape)
    for radius in range(max_radius + 1):
        local = dilate_mask(mask, radius)
        if float(energy[local].sum()) / total >= fraction:
            return float(radius)
    return float(max_radius)


def evaluate_shocks(model, static_raw, stats, data_phys, doy_all, times, lats, lons) -> None:
    """Run the two exact contract shocks against one real three-day input window."""
    date_strings = np.datetime_as_string(times, unit="D")
    matches = np.flatnonzero(date_strings == "2020-07-15")
    if not len(matches):
        print("\nShock evaluation: SKIP — 2020-07-15 is not present in the dataset")
        return
    last_input_idx = int(matches[0])
    if last_input_idx < 2:
        print("\nShock evaluation: SKIP — not enough prior days for three-day history")
        return

    history = data_phys[last_input_idx - 2 : last_input_idx + 1].copy()
    doy = float(doy_all[last_input_idx])
    base = rollout_10_days(model, history, static_raw, stats, doy, steps=STEPS).cpu().numpy()
    zero = rollout_10_days(model, history.copy(), static_raw, stats, doy, steps=STEPS).cpu().numpy()
    print(f"\nZero perturbation max |difference|: {np.max(np.abs(zero - base)):.8g}")

    bay = box_mask(lats, lons, (8.0, 20.0), (82.0, 94.0))
    arabian = box_mask(lats, lons, (8.0, 20.0), (65.0, 75.0))
    warm_ocean = np.zeros((6, *bay.shape), dtype=np.float32)
    warm_ocean[0, bay] = 3.0
    warm_ocean[5, bay] = 0.15 * history[-1, 5, bay]
    monsoon_stall = np.zeros((6, *arabian.shape), dtype=np.float32)
    monsoon_stall[2, arabian] = -0.40 * history[-1, 2, arabian]
    monsoon_stall[3, arabian] = -0.40 * history[-1, 3, arabian]

    channel_std = np.asarray(stats["std"], dtype=np.float64).reshape(1, 6, 1, 1)
    for label, perturbation, source_mask in (
        ("warm_ocean", warm_ocean, bay),
        ("monsoon_stall", monsoon_stall, arabian),
    ):
        shocked_history = history.copy()
        shocked_history[-1] += perturbation
        shocked = rollout_10_days(model, shocked_history, static_raw, stats, doy, steps=STEPS).cpu().numpy()
        diff = shocked - base
        response_max = float(np.max(np.abs(diff)))
        max_std = float(np.max(np.abs(diff) / channel_std))
        near = dilate_mask(source_mask, 3)
        day1_energy = np.sum(diff[0].astype(np.float64) ** 2, axis=0)
        local_fraction = float(day1_energy[near].sum() / max(day1_energy.sum(), 1e-18))
        day4_energy = np.sum(diff[3].astype(np.float64) ** 2, axis=0)
        day8_energy = np.sum(diff[7].astype(np.float64) ** 2, axis=0)
        outside4 = float(day4_energy[~near].sum() / max(day4_energy.sum(), 1e-18))
        outside8 = float(day8_energy[~near].sum() / max(day8_energy.sum(), 1e-18))
        r1 = energy_radius(diff[0], source_mask)
        r4 = energy_radius(diff[3], source_mask)
        r8 = energy_radius(diff[7], source_mask)
        rate14 = (r4 - r1) / 3.0
        rate48 = (r8 - r4) / 4.0
        print(f"\n{label} shock:")
        print(f"  maximum response: {response_max:.6g} physical units")
        print(f"  maximum standardized response: {max_std:.3f} std (limit 5)")
        print(f"  Day-1 response energy within 3 cells: {local_fraction * 100:.1f}% (target >=60%)")
        print(f"  Day-4 / Day-8 energy outside 3-cell buffer: {outside4 * 100:.1f}% / {outside8 * 100:.1f}%")
        print(f"  95%-energy radius D1/D4/D8: {r1:.1f}/{r4:.1f}/{r8:.1f} cells")
        print(f"  radius growth D1-D4 / D4-D8: {rate14:.2f}/{rate48:.2f} cells/day (limit <=8)")
        localized = response_max > 1e-6 and local_fraction >= 0.60
        outward = outside4 > 1e-6 and outside8 > 1e-6
        stable = max_std <= 5.0 and rate14 <= 8.0 and rate48 <= 8.0
        print(f"  localized non-zero response: {localized}")
        print(f"  outward response observed by D4 and D8: {outward}")
        print(f"  numerically stable: {stable}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="checkpoint to evaluate (defaults to the 3-day candidate if present, otherwise the serving checkpoint)",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = CANDIDATE_CKPT_PATH if CANDIDATE_CKPT_PATH.exists() else CKPT_PATH
    if not checkpoint_path.is_absolute():
        checkpoint_path = (ROOT / checkpoint_path).resolve()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing: {DATA_PATH}. Obtain the authorized dataset locally; never commit it."
        )
    if not CKPT_PATH.exists() or not STATIC_PATH.exists():
        raise FileNotFoundError("Required checkpoint or assets are missing")

    device = torch.device("cpu")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stats = checkpoint.get("stats")
    if not stats:
        raise ValueError("Checkpoint does not contain its normalization stats")
    model = MonsoonResUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    static_raw = np.load(STATIC_PATH).astype(np.float32)

    ds = xr.open_dataset(DATA_PATH)
    try:
        data_phys = ds["data"].values.astype(np.float32)
        doy_all = ds["time"].dt.dayofyear.values.astype(np.int32)
        times = ds["time"].values
        lats = ds["lat"].values.astype(np.float32)
        lons = ds["lon"].values.astype(np.float32)
    finally:
        ds.close()

    if data_phys.ndim != 4 or data_phys.shape[1:] != (6, 32, 36):
        raise ValueError(f"Unexpected data shape: {data_phys.shape}; expected [N,6,32,36]")
    if len(data_phys) <= TRAIN_DAYS:
        raise ValueError("Dataset does not include the 2020-2021 validation period")
    split_dates = (str(times[0])[:10], str(times[TRAIN_DAYS - 1])[:10], str(times[TRAIN_DAYS])[:10])
    if split_dates != ("2005-01-01", "2019-12-31", "2020-01-01"):
        raise ValueError(f"unexpected train/validation split dates: {split_dates}")

    train_phys = data_phys[:TRAIN_DAYS]
    train_doy = doy_all[:TRAIN_DAYS]
    clim_by_doy = compute_climatology(train_phys, train_doy)

    # Each window has 3 consecutive observed days, followed by 10 forecast targets.
    # Start at TRAIN_DAYS-3 so the first target is 2020-01-01 and the two preceding
    # input days may correctly come from the end of 2019.
    starts = np.arange(TRAIN_DAYS - HISTORY_DAYS, len(data_phys) - HISTORY_DAYS - STEPS + 1)
    if len(starts) == 0:
        raise ValueError("No complete held-out forecast windows are available")
    target_offsets = np.arange(HISTORY_DAYS, HISTORY_DAYS + STEPS)

    predictions, targets, persistence, climatology = [], [], [], []
    for batch_start in range(0, len(starts), 32):
        batch_starts = starts[batch_start : batch_start + 32]
        histories = np.stack([data_phys[i : i + HISTORY_DAYS] for i in batch_starts])
        x0 = torch.from_numpy(histories).to(device)
        doy = torch.from_numpy(doy_all[batch_starts + HISTORY_DAYS - 1].astype(np.float32)).to(device)
        pred = rollout_10_days(model, x0, static_raw, stats, doy, steps=STEPS).cpu().numpy()

        target_indices = batch_starts[:, None] + target_offsets[None, :]
        target_batch = data_phys[target_indices]
        persistence_batch = np.repeat(histories[:, -1:, ...], STEPS, axis=1)
        target_doy = doy_all[target_indices]
        clim_batch = np.stack(
            [np.stack([clim_by_doy[int(day)] for day in row], axis=0) for row in target_doy],
            axis=0,
        ).astype(np.float32)

        predictions.append(pred)
        targets.append(target_batch)
        persistence.append(persistence_batch)
        climatology.append(clim_batch)

    pred_all = np.concatenate(predictions, axis=0)
    true_all = np.concatenate(targets, axis=0)
    pers_all = np.concatenate(persistence, axis=0)
    clim_all = np.concatenate(climatology, axis=0)

    finite = bool(np.all(np.isfinite(pred_all)))
    minimum_rain = float(pred_all[:, :, 1].min())
    minimum_tcwv = float(pred_all[:, :, 5].min())
    print(f"Validation windows: {len(starts)}")
    print(f"Model history days: {model.history_days}")
    print(f"All forecast values finite: {finite}")
    print(f"Minimum precipitation: {minimum_rain:.6f} mm/day")
    print(f"Minimum TCWV: {minimum_tcwv:.6f} kg/m²")
    if not finite or minimum_rain < 0.0 or minimum_tcwv < 0.0:
        raise AssertionError("Forecast physical invariants failed")

    print(f"\nEvaluating checkpoint: {checkpoint_path}")
    print("\n=== Official lead-time benchmark (latitude-weighted RMSE) ===")
    print("Skill definition used here: 1 - model_RMSE / min(persistence_RMSE, climatology_RMSE).")
    print("The contract specifies the better-baseline comparison but does not publish a separate aggregation formula.")
    day4_beats_persistence = True
    skill_means = []
    for lead in LEADS:
        k = lead - 1
        rmse_model = latitude_weighted_rmse(pred_all[:, k], true_all[:, k], lats)
        rmse_persistence = latitude_weighted_rmse(pers_all[:, k], true_all[:, k], lats)
        rmse_climatology = latitude_weighted_rmse(clim_all[:, k], true_all[:, k], lats)
        best_baseline = np.minimum(rmse_persistence, rmse_climatology)
        skill = np.where(best_baseline > 1e-12, 1.0 - rmse_model / best_baseline, np.nan)
        skill_means.append(float(np.nanmean(skill)))
        if lead == 4:
            day4_beats_persistence = bool(np.all(rmse_model < rmse_persistence))

        print(f"\nDay {lead}:")
        print("channel | model RMSE | persistence RMSE | climatology RMSE | best-baseline skill")
        for c, name in enumerate(stats["channels"]):
            print(
                f"{name:>7s} | {rmse_model[c]:10.4f} | {rmse_persistence[c]:16.4f} | "
                f"{rmse_climatology[c]:16.4f} | {skill[c]:19.4f}"
            )
        print(f"Mean channel skill: {skill_means[-1]:.4f}")
        csi = heavy_rain_csi(pred_all[:, k, 1], true_all[:, k, 1])
        print(f"Heavy-rain CSI (tp >= {RAIN_THRESHOLD_MM_DAY:.1f} mm/day): {csi:.4f}")

    print(f"\nDay-4 weighted RMSE beats persistence on all six channels: {day4_beats_persistence}")
    print(f"Mean of per-channel skill across requested leads: {float(np.nanmean(skill_means)):.4f}")

    evaluate_shocks(model, static_raw, stats, data_phys, doy_all, times, lats, lons)


if __name__ == "__main__":
    main()
