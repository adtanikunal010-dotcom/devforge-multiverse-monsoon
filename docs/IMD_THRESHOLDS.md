# IMD-Grounded Impact Alert Thresholds

These rules are for the Member 2 dashboard impact panel. Use the API's physical units and canonical channel names from `R1_CONTRACT.md`; do not apply `assets/stats.json` normalization to alert calculations.

| Alert | Rule | Unit / window | Display guidance |
|---|---|---|---|
| Heatwave | `t2m > 313.15 K` | Daily 2 m temperature | Equivalent to above 40 °C. Count cells meeting the condition on the selected forecast day. |
| Heavy rainfall / flood risk | `tp > 64.5 mm/day` | Daily precipitation | Count cells meeting the condition on the selected forecast day. |
| Extreme cloudburst | `tp > 115.5 mm/day` | Daily precipitation | A higher-severity subset of heavy rainfall; show the extreme label instead of double-counting it as a separate affected cell. |
| Kharif agricultural drought stress | `sum(tp over forecast days 1–10) < 25 mm` **and** `mean(t2m over forecast days 1–10) > 308.15 K` | 10-day precipitation total and 10-day mean temperature | Equivalent temperature condition is above 35 °C. Both conditions must hold for a cell to be flagged. |

## Implementation notes

- The contract defines `tp` as daily accumulated precipitation on the wire, in `mm/day`; sum its ten forecast values to evaluate the 10-day drought condition (reported as a 10-day total in mm).
- `t2m` is in kelvin. Convert only for labels if desired: `°C = K - 273.15`. Compare using kelvin thresholds above.
- The comparisons are strict (`>` and `<`) as written. A value equal to a threshold does not trigger that rule.
- For the heatwave, heavy rainfall, and cloudburst counters, evaluate the selected day. The drought stress counter uses all ten forecast days.
- The underlying heavy-rain threshold is `tp > 64.5 mm/day`. For mutually exclusive dashboard counters, classify `64.5 < tp ≤ 115.5 mm/day` as heavy-rain-only and `tp > 115.5 mm/day` as cloudburst. The two displayed buckets together cover all heavy-rain cells without counting cloudburst cells twice.
- These cutoffs are the project playbook's specified IMD-grounded operational rules for this prototype. They are alert heuristics for the demo, not official warnings issued by IMD.
