"""
data_loader.py
Topic: pandas, numpy, StandardScaler, train_test_split, VarianceThreshold

What it does:
1. Loads CICIDS CSV file(s)
2. Preserves the Label column
3. Removes non-numeric feature columns (except Label)
4. Converts labels to binary:
       BENIGN -> 0
       ATTACK -> 1
5. Removes inf/NaN values
6. Clips extreme outliers
7. Removes zero-variance features
8. Standardizes features
9. Splits into train/test sets
10. Saves processed artifacts
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from config import DATA_RAW, DATA_PROCESSED, TEST_SIZE, RANDOM_STATE


def load_raw():
    """
    Load all CSV files from DATA_RAW and concatenate them.
    """
    files = glob.glob(os.path.join(DATA_RAW, "*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_RAW}\n"
            "Download CICIDS2017 and place the CSV files in this folder."
        )

    print(f"Found {len(files)} CSV file(s). Loading...")

    dfs = []
    for file in files:
        print(f"Loading: {os.path.basename(file)}")
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    return df


def clean(df):
    """
    Clean the dataset:
    - Ensure Label column exists
    - Remove non-numeric columns except Label
    - Encode Label to binary
    - Remove inf and NaN
    """

    # ------------------------------------------------------------------
    # 1. Ensure Label column exists
    # ------------------------------------------------------------------
    if "Label" not in df.columns:
        raise ValueError(
            "Label column not found in dataset.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # 2. Remove non-numeric columns except Label
    # ------------------------------------------------------------------
    non_numeric = [
        col
        for col in df.select_dtypes(exclude=[np.number]).columns.tolist()
        if col != "Label"
    ]

    if non_numeric:
        print(f"Dropping non-numeric: {non_numeric}")
        df.drop(columns=non_numeric, inplace=True, errors="ignore")

    # ------------------------------------------------------------------
    # 3. Encode labels
    # ------------------------------------------------------------------
    # Numeric labels:
    #   0 -> benign
    #   non-zero -> attack
    #
    # String labels:
    #   BENIGN / NORMAL / 0 -> 0
    #   anything else -> 1
    if pd.api.types.is_numeric_dtype(df["Label"]):
        df["Label"] = (df["Label"] != 0).astype(int)
    else:
        label_series = (
            df["Label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        benign_values = {"benign", "normal", "0"}

        df["Label"] = (~label_series.isin(benign_values)).astype(int)

    print(f"Label distribution: {df['Label'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 4. Replace inf values with NaN
    # ------------------------------------------------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ------------------------------------------------------------------
    # 5. Drop rows containing NaN
    # ------------------------------------------------------------------
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)

    print(f"Removed {rows_before - rows_after} rows containing NaN/inf.")
    print(f"Remaining rows: {rows_after}")

    # ------------------------------------------------------------------
    # 6. Reset index
    # ------------------------------------------------------------------
    df.reset_index(drop=True, inplace=True)

    return df


def prepare():
    """
    Full preprocessing pipeline.
    """
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and clean data
    # ------------------------------------------------------------------
    df = load_raw()
    df = clean(df)

    # ------------------------------------------------------------------
    # Separate features and labels
    # ------------------------------------------------------------------
    X = df.drop(columns=["Label"]).to_numpy(dtype=np.float64)
    y = df["Label"].to_numpy(dtype=np.int64)

    print(f"Initial feature shape: {X.shape}")

    # ------------------------------------------------------------------
    # Fill any remaining NaN with column median
    # ------------------------------------------------------------------
    if np.isnan(X).any():
        print("Filling remaining NaN values with column medians...")
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # ------------------------------------------------------------------
    # Clip extreme outliers to 1st and 99th percentiles
    # ------------------------------------------------------------------
    print("Clipping extreme values (1st–99th percentile)...")
    p01 = np.percentile(X, 1, axis=0)
    p99 = np.percentile(X, 99, axis=0)
    X = np.clip(X, p01, p99)

    print(
        f"Any inf: {np.any(np.isinf(X))} | "
        f"Any NaN: {np.any(np.isnan(X))}"
    )

    # ------------------------------------------------------------------
    # Remove zero-variance features
    # ------------------------------------------------------------------
    print("Removing zero-variance features...")
    selector = VarianceThreshold(threshold=0.0)
    X = selector.fit_transform(X)

    print(f"Features after variance filter: {X.shape[1]}")

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Train/Test Split
    # ------------------------------------------------------------------
    print("Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ------------------------------------------------------------------
    # Save processed artifacts
    # ------------------------------------------------------------------
    print("Saving processed files...")

    joblib.dump(
        (X_train, X_test, y_train, y_test),
        os.path.join(DATA_PROCESSED, "split.pkl")
    )

    joblib.dump(
        scaler,
        os.path.join(DATA_PROCESSED, "scaler.pkl")
    )

    joblib.dump(
        selector,
        os.path.join(DATA_PROCESSED, "selector.pkl")
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")
    print(f"Attack ratio in train set: {y_train.mean():.2%}")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


def load_processed():
    """
    Load previously saved train/test split.
    """
    path = os.path.join(DATA_PROCESSED, "split.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Processed data not found.\n"
            "Run data_loader.prepare() first."
        )

    return joblib.load(path)


if __name__ == "__main__":
    prepare()