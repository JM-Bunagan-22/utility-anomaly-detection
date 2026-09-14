import pandas as pd

from anomaly_detection import Z_SCORE_THRESHOLD, flag_isolation_forest, flag_zscore
from features import build_features


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


def test_flag_zscore_catches_a_clear_spike():
    # 20 flat days, then one huge spike — the spike should clear the
    # z-score threshold now that it no longer inflates its own baseline.
    values = [10] * 20 + [100]
    df = flag_zscore(build_features(_daily(values)))
    spike = df.iloc[-1]
    assert spike["flag_zscore"]


def test_flag_zscore_uses_threshold_when_baseline_has_variance():
    values = [8, 12, 9, 11, 10, 9, 11, 100]
    df = flag_zscore(build_features(_daily(values)))
    spike = df.iloc[-1]
    assert spike["flag_zscore"]
    assert spike["zscore"] > Z_SCORE_THRESHOLD


def test_flag_zscore_leaves_flat_series_unflagged():
    df = flag_zscore(build_features(_daily([10] * 15)))
    assert not df["flag_zscore"].any()


def test_flag_isolation_forest_adds_expected_columns():
    values = [10, 11, 9, 10, 12, 8, 10, 11, 9, 30, 10, 11]
    df = flag_isolation_forest(build_features(_daily(values)))
    assert "flag_isoforest" in df.columns
    assert "isoforest_score" in df.columns
    assert df["flag_isoforest"].dtype == bool
