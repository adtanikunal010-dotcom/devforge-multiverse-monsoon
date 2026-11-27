# Multiverse of Monsoon
## Project Brief

### Page 1 — Forecasting Engine

#### A physics-aware view of India's monsoon

Multiverse of Monsoon is designed to explore ten-day weather evolution over a 32 × 36 spatial grid covering India. The shared data contract contains near-surface temperature (`t2m`), precipitation (`tp`), eastward and northward wind (`u10`, `v10`), mean sea-level pressure (`msl`), and total column water vapour (`tcwv`). Inputs and outputs use physical units.

#### Residual U-Net and geographic conditioning

The model is an encoder-decoder residual U-Net (`MonsoonResUNet`, 510,470 parameters, 1.96 MiB checkpoint in `weights/monsoon_unet.pt`) that predicts daily normalized deltas conditioned on a land-sea mask, normalized elevation (`assets/static_maps.npy`), and seasonal day-of-year encoding. A damped 3-day temporal tendency blends `init_states[-3]` and `init_states[-2]` into the initial state before the 10-day autoregressive rollout.

#### Anti-blur and physical guardrails

Training uses a chronological split of **5,478 train days** (2005–2019) and **731 held-out validation days** (2020-01-01 to 2021-12-31, yielding **721 rolling 10-day windows**), with a 3-step rollout loss and 1.5x precipitation weighting. At inference, `tp` and `tcwv` are clamped to `>= 0.0` at every rollout step, non-finite values are guarded, and late-lead (`Day 6+`) spatial eddy-diffusion damping keeps Day-8 spatial variance ratios within the **0.40–2.50** anti-blur band across all six channels (`t2m` 1.059, `tp` 0.794, `u10` 1.391, `v10` 1.564, `msl` 1.626, `tcwv` 1.220).

#### Held-out benchmark skill (`assets/benchmark_results.json`)

- **Day-1 cos(lat)-weighted RMSE & skill vs. best baseline**: `tp` 5.1556 mm/day (+18.50%), `msl` 168.28 Pa (+15.17%), `v10` 0.9811 m/s (+11.65%), `u10` 0.9769 m/s (+10.41%), `tcwv` 3.3750 kg/m² (+9.92%), `t2m` 0.9704 K (+5.46%); Heavy-Rain CSI (`>= 10 mm/day`) = **0.4632** vs. Persistence 0.4290 and Climatology 0.2593.
- **Day-4 performance**: Beats Persistence across **all 6 variables** (`tp` 6.9456 vs. 9.2488 mm/day, `t2m` 1.8879 vs. 1.9183 K, `u10` 1.7797 vs. 2.0166 m/s, `v10` 1.7374 vs. 1.9453 m/s, `msl` 278.23 vs. 341.63 Pa, `tcwv` 6.5345 vs. 7.5082 kg/m²), with Day-4 Heavy-Rain CSI = **0.2672** (beating both Persistence 0.2176 and Climatology 0.2597).
- **Day-8 performance**: Beats Persistence across **all 6 variables** (`tp` 7.3823 vs. 9.3623 mm/day, `msl` 321.66 vs. 378.91 Pa, `tcwv` 7.9660 vs. 8.3519 kg/m², `u10` 2.1300 vs. 2.2265 m/s, `v10` 1.9883 vs. 1.9897 m/s; `t2m` 2.4109 vs. 2.2484 K), with Day-8 Heavy-Rain CSI = **0.2273** vs. Persistence 0.2033.

#### Counterfactual causality and runtime verification

- **Zero-shock guarantee**: `null` perturbation max difference = **0.0** (bit-for-bit identical to unperturbed baseline).
- **Warm Ocean (`+3 K t2m`, `+15% tcwv`)**: **95.01%** Day-1 response energy within 3 cells, max response 1.11 std, driving +1.57 mm/day (Day 4) and +2.20 mm/day (Day 8) mean `|Δtp|`.
- **Monsoon Stall (`-40% u10 & v10`)**: **98.17%** Day-1 response energy within 3 cells, max response 1.34 std, driving 0.90 mm/day (Day 4) and 0.83 mm/day (Day 8) mean `|Δtp|`.
- **Offline CPU & QA**: ~29–36 ms model inference (~84 ms end-to-end API response), ~0.27 GB server RAM, zero external runtime network requests, and **87/87** checks passing in `tests/verify_all.py` (0 FAIL, 0 WARN, 0 SKIP).

### Page 2 — Decisions and Experiments

#### Interactive Multiverse Dashboard

The offline dashboard (`dashboard/index.html` and `dashboard/indra3d.html`) provides synchronized baseline, shocked, and difference views across the 10-day horizon using local HTML, CSS, and JavaScript (with Three.js r187 bundled locally) and zero external runtime network requests.

#### Indian flood and drought decision support

The impact panel evaluates predicted grids against transparent physical thresholds: heatwave above 313.15 K, heavy daily rainfall above 64.5 mm/day, and extreme cloudburst rainfall above 115.5 mm/day. Kharif agricultural drought stress is flagged when ten-day precipitation is below 25 mm and ten-day mean temperature is above 308.15 K (detailed in `docs/IMD_THRESHOLDS.md`). These prototype alerts support scenario exploration and are not official IMD warnings.

#### Offline demonstration and readiness

The local presentation flow starts `serve.py --port 8000` (exposing `/health`, `/simulate`, `/forecast`, and `/terrain`), opens the dashboard, and compares baseline and perturbed scenarios. Held-out Day-1/4/8 skill, Heavy-Rain CSI, Day-8 anti-blur variance ratios, counterfactual shock causality, CPU latency, and RAM checks are all verified locally.

#### Project outcome

The project unifies learned 10-day monsoon dynamics, geographic conditioning, controllable counterfactual shock experiments, and interpretable agro-climatic thresholds in a compact (< 5 MB weights), 100% offline environment.
