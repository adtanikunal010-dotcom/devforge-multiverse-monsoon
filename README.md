# Multiverse of Monsoon

Physics-aware climate digital twin for 10-day monsoon forecasting and counterfactual flood, heatwave, and Kharif drought analysis over India.

> **Integration status:** The checked-in `serve.py` is still a persistence placeholder. The `train/`, `weights/`, `dashboard/`, and `tests/` deliverables are not present in this checkout yet. Do not treat this revision as a validated submission or demo; the model, UI, and checks must be integrated before judging.

## Run the offline demo

From the repository root:

1. Install the project dependencies in a Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Start the local API:

   ```bash
   python serve.py --port 8000
   ```

3. Open `dashboard/index.html` in a browser. The dashboard is designed to run without external services and sends simulation requests to `http://localhost:8000`.

The browser dashboard is supplied by the frontend workstream. If `dashboard/index.html` is not present yet, wait for the Member 2 integration before the demo.

## Architecture

The API contract accepts three consecutive daily states on a 32 × 36 India grid in the canonical physical-unit channel order `t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`. The target forecasting design uses residual daily-delta prediction, static land-sea and elevation conditioning, and multi-step rollout. Spatial-gradient loss is part of the planned anti-blur approach. Counterfactual perturbations are physical-unit deltas to the final input state; a zero perturbation must reproduce the baseline forecast. These are design requirements, not verified capabilities of the current placeholder server.

The API contract, grid coordinates, channel units, and request/response schemas are in [`R1_CONTRACT.md`](R1_CONTRACT.md). Threshold definitions for the impact alerts are in [`docs/IMD_THRESHOLDS.md`](docs/IMD_THRESHOLDS.md).

## Submission safety and readiness

- Run with Wi-Fi disabled. Do not call external weather APIs or download weights at runtime; required weights and runtime assets must be committed in the repository.
- Do not add raw historical weather archives, hidden evaluation data, or lookup tables of historical dates. Keep the repository below 2 GB.
- The judging container mounts the repository read-only. Do not write logs, caches, or temporary files into it at runtime; use `/tmp` only when required.
- The readiness bar is valid finite output for all eight cases, non-negative rainfall, exact zero-shock equivalence, a sensible response to a non-zero shock, Day 4 skill better than persistence, and CPU rollout under five seconds.
- Export the Project Brief as exactly two pages. The current brief marks validation as pending where results have not been measured; do not replace that status with unverified claims.
- The field manual specifies a hard code freeze at 9:00 AM and says to stop Git pushes by 8:55 AM. Follow the event's announced freeze and do not push afterward.

## Validation

With the server running, validate the API contract:

```bash
python -m monsoon_r1.schema_validator http://localhost:8000
```

Run the repository scan and automated tests when the QA workstream has added them:

```bash
python -m monsoon_r1.repo_scan.scan .
python -m pytest tests/test_contract.py
python tests/benchmark.py
```

For the submission rehearsal, disable Wi-Fi and verify the cold start, API response, and dashboard locally. The server must bind to `0.0.0.0`; the browser UI and runtime assets must not require an internet connection.
