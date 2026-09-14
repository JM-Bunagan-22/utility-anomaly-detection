import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, build_features


def _daily(values, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "total_kwh": values,
            "avg_power": [v / 24 for v in values],
            "max_power": [max(v / 12, 0.1) for v in values],
            "min_power": [0.05] * len(values),
        }
    )


def test_all_feature_columns_present():
    df = build_features(_daily([10, 11, 9, 10, 12, 8, 10, 11]))
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_rolling_mean_excludes_current_day():
    # A flat baseline followed by one spike: the spike day's rolling mean
    # must reflect only the flat days before it, not the spike itself.
    values = [10, 10, 10, 10, 10, 10, 10, 100]
    df = build_features(_daily(values))
    spike_row = df.iloc[-1]
    assert spike_row["rolling_mean_7d"] == 10
    assert spike_row["day_over_day_change"] == 90


def test_first_day_has_no_baseline():
    df = build_features(_daily([10, 11, 12]))
    first = df.iloc[0]
    assert np.isnan(first["rolling_mean_7d"])
    assert first["rolling_std_7d"] == 0
    assert first["pct_change_from_rolling"] == 0


def test_is_weekend_flags_saturday_and_sunday():
    # 2024-01-01 is a Monday, so day 5 (Sat) and day 6 (Sun) are the weekend.
    df = build_features(_daily([10] * 8))
    assert list(df["is_weekend"]) == [0, 0, 0, 0, 0, 1, 1, 0]


def test_load_factor_is_bounded_ratio():
    df = build_features(_daily([10, 12, 8, 15]))
    assert (df["load_factor"] >= 0).all()
