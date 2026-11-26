# Multiverse of Monsoon
## Project Brief

### Page 1 — Forecasting Engine

#### A physics-aware view of India's monsoon

Multiverse of Monsoon is designed to explore ten-day weather evolution over a 32 × 36 grid covering India. The shared data contract contains near-surface temperature (`t2m`), precipitation (`tp`), eastward and northward wind (`u10`, `v10`), mean sea-level pressure (`msl`), and total column water vapour (`tcwv`). Inputs and outputs use physical units.

#### Residual U-Net and geographic conditioning

The target model is an encoder-decoder residual U-Net that predicts daily changes from consecutive weather states. A land-sea mask and normalized elevation provide static context for coastlines, terrain, and orographic rainfall patterns. The model applies each predicted residual to the current state to form a ten-day autoregressive rollout.

#### Anti-blur and physical guardrails

The anti-blur approach combines residual delta prediction with a spatial-gradient loss to preserve local structure, alongside a multi-step rollout loss to reduce accumulated drift. The training source weights precipitation loss and clamps predicted rain at zero. It uses a chronological split, with later years reserved for validation. The committed checkpoint loads strictly into the current model implementation.

#### Counterfactual consistency

The same initial conditions can be run without a shock or with a localized perturbation, such as warmer Bay of Bengal temperatures or weaker Arabian Sea winds. A zero-valued perturbation must match the unperturbed run. This zero-shock guarantee makes the scenario difference attributable to the supplied perturbation.

#### Validation results and known failure modes

The repository verifier reported 31 passed, 0 failed, and 3 skipped. The skips were the official repository scan and schema validator (their `monsoon_r1` tools are not installed here) and process-RAM inspection (blocked by macOS permissions). The separate local benchmark measured 23.2 ms median and 23.9 ms maximum inference over three ten-day calls, with 0.243 GB observed peak server RSS. These are workstation measurements, not results from the judges' four-core sandbox. The source training dataset is absent, so the checkpoint could not be regenerated here and held-out forecast skill versus persistence remains unverified. No forecast-skill or qualification claim is made.

### Page 2 — Decisions and Experiments

#### Interactive Multiverse Dashboard

The offline dashboard shows three synchronized views: the baseline forecast, a shocked forecast, and their cell-by-cell difference. A day slider and variable selector let users inspect precipitation, temperature, wind, pressure, and moisture across the ten-day horizon. Preset and map-based shocks support counterfactual exploration while the initial state stays fixed. The interface uses local HTML, CSS, and JavaScript without external requests; browser-level visual verification remains part of the Wi-Fi-off rehearsal.

#### Indian flood and drought decision support

The impact panel is designed to count cells against transparent physical thresholds: heatwave above 313.15 K, heavy daily rainfall above 64.5 mm/day, and extreme cloudburst rainfall above 115.5 mm/day. Kharif agricultural drought stress is flagged when ten-day precipitation is below 25 mm and ten-day mean temperature is above 308.15 K. The detailed rules are in `docs/IMD_THRESHOLDS.md`.

These prototype alerts support exploration of where hazards may coincide and how a scenario changes the affected area. They are not official warnings or a substitute for local forecasts and expert assessment.

#### Offline demonstration and readiness

The presentation flow starts the local `/simulate` API, opens the dashboard, and compares baseline and perturbed scenarios. Contract, zero-shock, physical-guard, and local CPU/RSS checks pass. The offline cold-start rehearsal and held-out forecast-skill comparison still need to be completed before making a final submission-readiness claim.

#### Project outcome

The project brings together learned weather evolution, geographic context, controllable counterfactual experiments, and interpretable alert thresholds. The goal is a compact environment for examining how monsoon scenarios could affect flood, heat, and crop-water stress across India.
