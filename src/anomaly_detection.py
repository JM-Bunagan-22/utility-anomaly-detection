"""
Flag anomalous consumption days using two approaches:
  1. Statistical: rolling z-score threshold
  2. ML: Isolation Forest (unsupervised)

Run after data_loader.py has produced data/daily_consumption.csv.
"""

import os
import pandas as pd
from sklearn.ensemble import IsolationForest

from features import build_features, FEATURE_COLUMNS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DAILY_PATH = os.path.join(DATA_DIR, "daily_consumption.csv")
OUT_PATH = os.path.join(DATA_DIR, "flagged_consumption.csv")

Z_SCORE_THRESHOLD = 2.5
ISOLATION_FOREST_CONTAMINATION = 0.05  # assume ~5% of days are anomalous


def flag_zscore(df: pd.DataFrame) -> pd.DataFrame:
    z = (df["total_kwh"] - df["rolling_mean_7d"]) / df["rolling_std_7d"].replace(0, pd.NA)
    df["zscore"] = z.fillna(0)
    df["flag_zscore"] = df["zscore"].abs() > Z_SCORE_THRESHOLD
    return df


def flag_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    model = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=42,
    )
    X = df[FEATURE_COLUMNS].fillna(0)
    df["flag_isoforest"] = model.fit_predict(X) == -1  # -1 = outlier
    df["isoforest_score"] = model.decision_function(X)  # lower = more anomalous
    return df


def run():
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    df = build_features(daily)
    df = flag_zscore(df)
    df = flag_isolation_forest(df)

    df["flagged_by_both"] = df["flag_zscore"] & df["flag_isoforest"]

    df.to_csv(OUT_PATH, index=False)

    print(f"Total days analyzed: {len(df)}")
    print(f"Flagged by z-score: {df['flag_zscore'].sum()}")
    print(f"Flagged by Isolation Forest: {df['flag_isoforest'].sum()}")
    print(f"Flagged by both methods (highest confidence): {df['flagged_by_both'].sum()}")
    print(f"Saved results to {OUT_PATH}")

    return df


if __name__ == "__main__":
    run()
