# Multiverse of Monsoon

Physics-aware climate digital twin for 10-day monsoon forecasting and counterfactual flood, heatwave, and Kharif drought analysis over India.

> **Integration status:** The inference model, checkpoint, offline dashboard, and QA suite are present. Local contract, shock, physics, latency, and memory checks pass. Held-out forecast skill is not verified because the source training dataset is not included, so the weights could not be regenerated or compared with persistence in this checkout.

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

The API accepts three consecutive daily states on a 32 × 36 India grid in the canonical physical-unit channel order `t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`. The residual U-Net predicts normalized daily deltas, uses static land-sea and elevation conditioning, and rolls forward autoregressively. Training includes multi-step loss, precipitation weighting, and spatial-gradient loss. The inference rollout clamps precipitation at zero. Counterfactual perturbations are physical-unit deltas to the final input state; a zero perturbation reproduces the baseline. The dashboard compares baseline, shocked, and difference maps.

The API grid, units, request/response schemas, and channel order are defined in [`R1_CONTRACT.md`](R1_CONTRACT.md). Alert thresholds are documented in [`docs/IMD_THRESHOLDS.md`](docs/IMD_THRESHOLDS.md).

## Validation

Start the API in one terminal and run the repository QA verifier in another:

```bash
PYTHONDONTWRITEBYTECODE=1 python serve.py --port 8000
PYTHONDONTWRITEBYTECODE=1 python tests/verify_all.py --url http://localhost:8000
```

The single `tests/verify_all.py` runner checks repository structure and offline-safety rules, the API schema and physical invariants, zero-shock equality, non-zero shock sensitivity, five consecutive CPU rollouts, and peak server RAM. It also runs the official schema and repository scans when those tools are installed. Results are reported in the Project Brief. These are local measurements, not results from the judges' four-core sandbox. A held-out Day 4 skill comparison against persistence remains pending because `data/india_climate_train.nc` is not in this repository.

## Submission safety

- Rehearse with Wi-Fi disabled. The UI has no external requests; do not call external weather APIs or download weights at runtime.
- Keep the committed model weights and runtime assets inside the repository. Do not add raw historical archives, hidden evaluation data, or hard-coded historical-date lookup tables. Keep the repository below 2 GB.
- The judging container mounts the repository read-only. Do not write logs, caches, or temporary files there at runtime; use `/tmp` only if required.
- Before submission, complete the offline cold-start rehearsal and verify all eight evaluation cases, the Day 4 skill gate, zero-shock behavior, physical guards, CPU latency, and the two-page Project Brief.
- The field manual specifies a 9:00 AM hard code freeze and says to stop pushes by 8:55 AM. Follow the event's announced freeze; do not push afterward.
