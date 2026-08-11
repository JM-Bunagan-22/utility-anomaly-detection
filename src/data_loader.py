"""
Download and prepare the UCI Individual Household Electric Power Consumption dataset.

Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
"""

import os
import zipfile
import pandas as pd
import requests

RAW_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_PATH = os.path.join(DATA_DIR, "household_power.zip")
TXT_PATH = os.path.join(DATA_DIR, "household_power_consumption.txt")
DAILY_OUT = os.path.join(DATA_DIR, "daily_consumption.csv")


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TXT_PATH):
        print("Raw data already present, skipping download.")
        return
    print("Downloading dataset...")
    resp = requests.get(RAW_URL, timeout=60)
    resp.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        f.write(resp.content)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    print("Downloaded and extracted.")


def load_and_clean():
    """Load the raw minute-level data and aggregate to daily consumption."""
    print("Loading raw data (this file is large, may take a minute)...")
    df = pd.read_csv(
        TXT_PATH,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    df = df.dropna(subset=["Global_active_power"])
    df["Global_active_power"] = df["Global_active_power"].astype(float)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df["date"] = df["datetime"].dt.date

    # Aggregate to daily totals — this is the level anomaly detection will run on
    daily = (
        df.groupby("date")
        .agg(
            total_kwh=("Global_active_power", lambda x: (x * (1 / 60)).sum()),  # minute readings -> kWh
            avg_power=("Global_active_power", "mean"),
            max_power=("Global_active_power", "max"),
            min_power=("Global_active_power", "min"),
            readings_count=("Global_active_power", "count"),
        )
        .reset_index()
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    daily.to_csv(DAILY_OUT, index=False)
    print(f"Saved daily consumption data to {DAILY_OUT} ({len(daily)} days)")
    return daily


if __name__ == "__main__":
    download_data()
    load_and_clean()
