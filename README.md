# Multiverse of Monsoon

Physics-aware climate digital twin for 10-day monsoon forecasting and counterfactual flood, heatwave, and Kharif drought analysis over India.

> **Integration status:** The inference model, trained checkpoint (`weights/monsoon_unet.pt`, 1.96 MiB), offline dashboard, held-out benchmark results (`assets/benchmark_results.json`), and QA suite are complete. All **87/87** checks in `tests/verify_all.py` pass (0 FAIL, 0 WARN, 0 SKIP) with zero external runtime network requests.

## Run the offline demo

From the repository root:

1. Create and populate a Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Start the local API:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python serve.py --port 8000
   ```

3. Open `dashboard/index.html` (or `dashboard/indra3d.html`) in a browser. The UI communicates only with `http://localhost:8000` and uses zero external CDN requests.

## Architecture

The API accepts three consecutive daily states on a 32 × 36 spatial grid over India in the canonical physical-unit channel order `t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`. `MonsoonResUNet` (510,470 parameters) predicts normalized daily deltas conditioned on static land-sea and elevation maps (`assets/static_maps.npy`) plus seasonal encoding. `serve.py` blends a damped 3-day temporal tendency from `init_states[-3]` and `init_states[-2]`, clamps `tp >= 0.0` and `tcwv >= 0.0` at every rollout step, applies gentle Day-6+ spatial eddy-diffusion damping on `u10`, and guarantees bit-for-bit zero-shock equality.

The API contract is defined in [`R1_CONTRACT.md`](R1_CONTRACT.md) and alert thresholds are documented in [`docs/IMD_THRESHOLDS.md`](docs/IMD_THRESHOLDS.md).

## Held-out benchmark results (`assets/benchmark_results.json`)

Evaluated chronologically on **5,478 train days** (2005–2019) and **731 validation days** (2020-01-01 to 2021-12-31, **721 rolling 10-day windows**):

- **Day-1 `cos(lat)`-weighted RMSE & skill vs. best baseline:** `tp` 5.1556 mm/day (+18.50%), `msl` 168.28 Pa (+15.17%), `v10` 0.9811 m/s (+11.65%), `u10` 0.9769 m/s (+10.41%), `tcwv` 3.3750 kg/m² (+9.92%), `t2m` 0.9704 K (+5.46%). Day-1 Heavy-Rain CSI (`>= 10 mm/day`) = **0.4632** vs. Persistence 0.4290 and Climatology 0.2593.
- **Day-4 performance:** Beats Persistence across **all 6 variables** (`tp` 6.9456 vs. 9.2488 mm/day, `t2m` 1.8879 vs. 1.9183 K, `u10` 1.7797 vs. 2.0166 m/s, `v10` 1.7374 vs. 1.9453 m/s, `msl` 278.23 vs. 341.63 Pa, `tcwv` 6.5345 vs. 7.5082 kg/m²), with Day-4 Heavy-Rain CSI = **0.2672** (beating both Persistence 0.2176 and Climatology 0.2597).
- **Day-8 performance:** Beats Persistence across `tp` (7.3823 vs. 9.3623 mm/day), `msl` (321.66 vs. 378.91 Pa), `tcwv` (7.9660 vs. 8.3519 kg/m²), `u10` (2.1300 vs. 2.2265 m/s), and `v10` (1.9883 vs. 1.9897 m/s), with Heavy-Rain CSI = **0.2273** vs. Persistence 0.2033 and all 6 Day-8 spatial variance ratios within **0.40–2.50** (`t2m` 1.059, `tp` 0.794, `u10` 1.391, `v10` 1.564, `msl` 1.626, `tcwv` 1.220).
- **Counterfactual shock causality:** Zero-shock max diff = **0.0**. Warm Ocean (`+3 K t2m`, `+15% tcwv`) = **95.01%** Day-1 local energy, +1.57 mm/day Day-4 & +2.20 mm/day Day-8 mean `|Δtp|`. Monsoon Stall (`-40% u10 & v10`) = **98.17%** Day-1 local energy, 0.90 mm/day Day-4 & 0.83 mm/day Day-8 mean `|Δtp|`.

## Validation

Start the API in one terminal and run the repository QA verifier in another:

```bash
PYTHONDONTWRITEBYTECODE=1 python serve.py --port 8000
PYTHONDONTWRITEBYTECODE=1 python tests/verify_all.py --url http://localhost:8000 --reference-dataset data/india_climate_train.nc
```

- **Runtime & offline metrics:** ~29–36 ms model inference (~84 ms end-to-end API response) on laptop CPU, ~0.27 GB server RAM, zero external runtime network requests, and **87/87** `tests/verify_all.py` checks passing.

## Submission safety

- Rehearse with Wi-Fi disabled. All UI assets and Three.js (`dashboard/vendor/three.global.js`) are bundled locally with no external requests.
- Keep committed model weights (`1.96 MiB < 5 MiB`) and runtime assets inside the repository, and exclude raw `.nc` archives from Git.
- The judging container mounts the repository read-only; do not write logs or caches into the repo at runtime.
