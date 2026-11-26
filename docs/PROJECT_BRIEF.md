# Multiverse of Monsoon
## Project Brief

### Page 1 — Forecasting Engine

#### A physics-aware view of India's monsoon

Multiverse of Monsoon is designed to explore ten-day weather evolution over a 32 × 36 grid covering India. The shared data contract contains near-surface temperature (`t2m`), precipitation (`tp`), eastward and northward wind (`u10`, `v10`), mean sea-level pressure (`msl`), and total column water vapour (`tcwv`). Inputs and outputs use physical units.

#### Residual U-Net and geographic conditioning

The target model is an encoder-decoder residual U-Net that predicts daily changes from consecutive weather states. A land-sea mask and normalized elevation provide static context for coastlines, terrain, and orographic rainfall patterns. The model applies each predicted residual to the current state to form a ten-day autoregressive rollout.

#### Anti-blur and physical guardrails

The planned anti-blur approach combines residual delta prediction with a spatial-gradient loss to preserve local structure, alongside a multi-step rollout loss to reduce accumulated drift. Precipitation must remain non-negative in physical units. Training is intended to use a chronological split, with later years reserved for validation, and additional loss weight on precipitation. These are design requirements; they are not yet confirmed as implemented in this checkout.

#### Counterfactual consistency

The same initial conditions can be run without a shock or with a localized perturbation, such as warmer Bay of Bengal temperatures or weaker Arabian Sea winds. A zero-valued perturbation must match the unperturbed run. This zero-shock guarantee makes the scenario difference attributable to the supplied perturbation.

#### Validation results and known failure modes

At brief preparation, this checkout still contains the starter persistence server and has no trained weights, training code, dashboard folder, or QA tests. The server repeats the last input state and adds perturbations directly; it does not demonstrate learned evolution, shock propagation, or a precipitation clamp. Forecast skill, spatial sharpness, zero-shock behavior, and CPU latency have not been validated here. No score or pass claim is made. These are integration blockers to resolve before submission.

### Page 2 — Decisions and Experiments

#### Interactive Multiverse Dashboard

The planned offline dashboard shows three synchronized views: the baseline forecast, a shocked forecast, and their cell-by-cell difference. A day slider and variable selector let users inspect precipitation, temperature, wind, pressure, and moisture across the ten-day horizon. Preset and map-based shocks support counterfactual exploration while the initial state stays fixed. The dashboard folder is not yet present in this checkout, so this remains a design target rather than a verified live interface.

#### Indian flood and drought decision support

The impact panel is designed to count cells against transparent physical thresholds: heatwave above 313.15 K, heavy daily rainfall above 64.5 mm/day, and extreme cloudburst rainfall above 115.5 mm/day. Kharif agricultural drought stress is flagged when ten-day precipitation is below 25 mm and ten-day mean temperature is above 308.15 K. The detailed rules are in `docs/IMD_THRESHOLDS.md`.

These prototype alerts support exploration of where hazards may coincide and how a scenario changes the affected area. They are not official warnings or a substitute for local forecasts and expert assessment.

#### Offline demonstration and readiness

The intended presentation flow starts the local `/simulate` API, opens the dashboard, and compares baseline and perturbed scenarios. Before the demo, the contract validator, repository scan, zero-shock check, physical guards, and CPU benchmark must pass in an offline cold-start rehearsal. Those results are pending integration and have not been established by this checkout.

#### Project outcome

The project brings together learned weather evolution, geographic context, controllable counterfactual experiments, and interpretable alert thresholds. The goal is a compact environment for examining how monsoon scenarios could affect flood, heat, and crop-water stress across India.
