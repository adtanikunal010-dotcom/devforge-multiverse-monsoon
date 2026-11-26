# Multiverse of Monsoon

Physics-aware climate digital twin for 10-day monsoon forecasting and counterfactual flood, heatwave, and Kharif drought analysis over India.

> **Integration status:** The inference model, checkpoint, offline dashboard, and QA suite are present. Local API, shock, and physical-invariant checks pass. Held-out forecast skill is not verified because the source training dataset is not included, so the weights could not be regenerated or compared with persistence in this checkout.

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

3. Open `dashboard/index.html` in a browser. The UI sends requests only to `http://localhost:8000` and uses a clearly labelled synthetic sample state until a real three-day initial state is connected.

## Architecture

The API accepts three consecutive daily states on a 32 × 36 India grid in the canonical physical-unit channel order `t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`. The residual U-Net predicts normalized daily deltas, uses static land-sea and elevation conditioning, and rolls forward autoregressively. Training includes multi-step loss, precipitation weighting, and spatial-gradient loss. The inference rollout clamps precipitation at zero. Counterfactual perturbations are physical-unit deltas to the final input state; a zero perturbation reproduces the baseline. The dashboard compares baseline, shocked, and difference maps with interactive Three.js 3D grid surfaces. Three.js r187 is bundled at `dashboard/vendor/three.global.js` under its MIT license. A compact local copy of the model land mask and elevation grid is in `dashboard/static-map-data.js`; both assets load from the repository, with no frontend CDN requests.

The API grid, units, request/response schemas, and channel order are defined in [`R1_CONTRACT.md`](R1_CONTRACT.md). Alert thresholds are documented in [`docs/IMD_THRESHOLDS.md`](docs/IMD_THRESHOLDS.md).

## Known limitations and failure modes

- **Forecast skill is unverified.** The held-out training dataset is not in this checkout, so the committed checkpoint could not be regenerated or compared against persistence and climatology. It may fail the Day 4 skill gate or perform poorly on unseen extremes; no qualification claim is made.
- **The dashboard starts from synthetic inputs.** Its built-in uniform sample is for exercising the interface and API. Impact counts from that sample are scenario examples, not observations or current warnings.
- **Impact alerts are prototype rules.** The thresholds are transparent demo criteria, not official IMD advisories. The India outline is simplified for map orientation.
- **Some validation remains environment-dependent.** The Wi-Fi-off clean-clone rehearsal passed on 2026-11-27. The verifier still skips the contest-provided schema/repository tools because `monsoon_r1` is unavailable here, and macOS denied process-RAM inspection. Official sandbox validation remains outstanding.

## Validation

Start the API in one terminal and run the repository QA verifier in another:

```bash
PYTHONDONTWRITEBYTECODE=1 python serve.py --port 8000
PYTHONDONTWRITEBYTECODE=1 python tests/verify_all.py --url http://localhost:8000
```

The single `tests/verify_all.py` runner checks repository structure and offline-safety rules, API schema and physical invariants, zero-shock equality, non-zero shock sensitivity, and five consecutive CPU rollouts. It attempts process-RAM measurement and runs the official schema/repository scans when those tools are available. On 2026-11-27, the clean-clone verifier reported 31 passed, 0 failed, and 3 skipped with Wi-Fi disabled. The separate workstation benchmark measured 23.2 ms median, 23.9 ms maximum inference, and 0.243 GB observed peak server RSS. These are local measurements, not results from the judges' four-core sandbox. A held-out Day 4 skill comparison against persistence remains pending because `data/india_climate_train.nc` is not in this repository.

## Submission safety

- Rehearse with Wi-Fi disabled. The UI has no external requests; do not call external weather APIs or download weights at runtime.
- Keep the committed model weights and runtime assets inside the repository. Do not add raw historical archives, hidden evaluation data, or hard-coded historical-date lookup tables. Keep the repository below 2 GB.
- The judging container mounts the repository read-only. Do not write logs, caches, or temporary files there at runtime; use `/tmp` only if required.
- Before submission, run the contest-provided schema/repository validators and verify the held-out Day 4 skill gate when the authorized dataset is available. The Wi-Fi-off clean-clone API rehearsal, zero-shock behavior, physical guards, CPU latency, and two-page Project Brief have been checked locally; final verification in the official sandbox is still required.
- The field manual specifies a 9:00 AM hard code freeze and says to stop pushes by 8:55 AM. Follow the event's announced freeze; do not push afterward.
