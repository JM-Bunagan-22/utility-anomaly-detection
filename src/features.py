"""
Feature engineering for anomaly detection on daily consumption data.
"""

import pandas as pd


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()

    # Rolling behavior — captures gradual drift vs. sudden change
    df["rolling_mean_7d"] = df["total_kwh"].rolling(window=7, min_periods=1).mean()
    df["rolling_std_7d"] = df["total_kwh"].rolling(window=7, min_periods=1).std().fillna(0)

    # Deviation from recent normal — the core anomaly signal
    df["pct_change_from_rolling"] = (
        (df["total_kwh"] - df["rolling_mean_7d"]) / df["rolling_mean_7d"].replace(0, pd.NA)
    ).fillna(0)

    # Day-over-day change — catches sudden drops (possible tampering) or spikes
    df["day_over_day_change"] = df["total_kwh"].diff().fillna(0)

    # Load factor — ratio of average to peak power; unusually flat or spiky
    # profiles can indicate meter issues or irregular usage
    df["load_factor"] = (df["avg_power"] / df["max_power"].replace(0, pd.NA)).fillna(0)

    # Calendar features — usage naturally varies by day of week
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


FEATURE_COLUMNS = [
    "total_kwh",
    "pct_change_from_rolling",
    "day_over_day_change",
    "load_factor",
    "is_weekend",
]
