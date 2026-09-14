# ⚡ Utility Consumption Anomaly Detection

Detecting suspicious or abnormal electricity consumption patterns using public household power-usage data — a public-data mirror of real-world utility fraud/anomaly detection workflows.

## Problem
Utility companies lose significant revenue to non-technical losses: meter tampering, unauthorized connections, and billing irregularities. Manually reviewing every account for suspicious patterns doesn't scale. This project builds a pipeline that flags days worth a closer look, based on consumption behavior alone.

## Approach
1. **Data**: [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) — minute-level power readings from a single household over ~4 years, aggregated to daily totals (`src/data_loader.py`).
2. **Feature engineering** (`src/features.py`): a 7-day rolling mean/std of the *prior* days (never the day being scored), day-over-day change, percent deviation from the rolling baseline, load factor (avg/peak power), and day-of-week/weekend flags.
3. **Anomaly detection** (`src/anomaly_detection.py`): compare a statistical rolling z-score threshold against an unsupervised ML model (Isolation Forest) trained on the engineered features, and flag the overlap as highest-confidence.
4. **Visualization** (`src/dashboard.py`): interactive Dash dashboard showing consumption trends with flagged days highlighted, so a reviewer can see *why* a day was flagged.

## Result
Analyzed 1,433 days of household electricity consumption. The rolling z-score method (threshold 2.5σ) flagged **98 days** as anomalous, Isolation Forest flagged **72 days (~5%)** — consistent with its configured contamination rate — and the two methods agreed on **35 days**, the highest-confidence flags. The remaining disagreement highlights a real difference in what each method catches: z-score only looks at `total_kwh` against its own recent history, so it also flags days that are just an unusually large single-day jump; Isolation Forest looks across all engineered features simultaneously (day-over-day change, load factor, weekend behavior), catching subtler multivariate anomalies — or missing a univariate spike whose other features still look "normal" for the day of week. This mirrors a real tradeoff in fraud/anomaly detection systems: simple rule-based flags are transparent and fast but noisier; ML models catch subtler multivariate patterns but still need review by an analyst before acting on the score alone. Days flagged by both are the natural starting point for that review.

*(An earlier version of the z-score baseline included each day's own value in the rolling mean/std used to judge it, which dampened every outlier's score and caused the method to flag 0 days. Fixed by excluding the current day from its own baseline — see `src/features.py`.)*

![Dashboard screenshot](assets/dashboard.png)

## Stack
Python · pandas · scikit-learn · Dash · Plotly · pytest

## Project Structure
```
utility-anomaly-detection/
├── data/
│   ├── household_power.zip        # raw UCI zip (checked in, ~20MB)
│   ├── daily_consumption.csv      # output of data_loader.py
│   └── flagged_consumption.csv    # output of anomaly_detection.py
├── src/
│   ├── data_loader.py       # download & clean the dataset
│   ├── features.py          # feature engineering
│   ├── anomaly_detection.py # detection models
│   └── dashboard.py         # Dash app
├── tests/              # unit tests (pytest)
├── assets/             # dashboard screenshot for this README
└── requirements.txt
```

## Setup
Pre-computed `data/*.csv` are already checked in, so you can go straight to the dashboard — or re-run the full pipeline yourself:
```bash
git clone https://github.com/JM-Bunagan-22/utility-anomaly-detection.git
cd utility-anomaly-detection
pip install -r requirements.txt

python src/data_loader.py       # downloads & prepares data (skips if already present)
python src/anomaly_detection.py # runs detection, saves flagged results
python src/dashboard.py         # launches Dash app at localhost:8050
```

## Configuration
Detection sensitivity is controlled by two constants at the top of `src/anomaly_detection.py`:
- `Z_SCORE_THRESHOLD` (default `2.5`) — how many standard deviations from the 7-day baseline counts as anomalous.
- `ISOLATION_FOREST_CONTAMINATION` (default `0.05`) — expected fraction of days that are anomalous.

Re-run `python src/anomaly_detection.py` after changing either to regenerate `data/flagged_consumption.csv` (and see the "Testing" note below for how the rolling-window edge cases are protected against regressions if you touch `src/features.py`).

## Output
`data/flagged_consumption.csv` has one row per day with the raw daily totals, every engineered feature, and the detection results a reviewer needs to see *why* a day was flagged:

| Column | Meaning |
| --- | --- |
| `zscore`, `flag_zscore` | z-score vs. the 7-day rolling baseline, and whether it crossed the threshold |
| `flag_isoforest`, `isoforest_score` | Isolation Forest outlier flag and raw decision score (lower = more anomalous) |
| `flagged_by_both` | agreement between both methods — the highest-confidence anomalies |

## Testing
```bash
pytest tests/
```
Unit tests cover the feature-engineering rolling windows (including the day-zero and zero-variance edge cases) and both flagging methods, using small synthetic series instead of the full dataset.

## Next Steps
- [x] Add screenshot of dashboard here
- [x] Fix the z-score baseline leaking each day's own value into its rolling mean/std
- [ ] Tune the z-score threshold and inspect a sample of the days it flags that Isolation Forest doesn't
- [ ] Add an IQR-based flagging method as a third, non-parametric point of comparison
- [ ] Compare model performance against a labeled anomaly set (if available)
- [ ] Extend to multiple households / accounts for a portfolio-level view
