#!/usr/bin/env python3
"""NOAA SST/ONI based LSTM forecasting example.

Supports two sources:
1) NOAA monthly SST index text file (sstoi.indices)
2) Local ONI Excel file (wide format: year + DJF..NDJ)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Avoid matplotlib cache permission warnings in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

NOAA_SST_INDEX_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
DEFAULT_ONI_EXCEL_PATH = (
    "/Users/kimtaeyoon/Library/CloudStorage/OneDrive-백운중학교/통합 문서1.xlsx"
)

ONI_SEASON_TO_MONTH = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,
}


@dataclass
class DatasetBundle:
    series: pd.Series
    scaler: MinMaxScaler
    train_scaled: np.ndarray
    val_scaled: np.ndarray
    test_scaled: np.ndarray


def load_noaa_sst_index(url: str) -> pd.DataFrame:
    """Load NOAA sstoi.indices format."""
    col_names = [
        "year",
        "month",
        "nino12",
        "nino12_anom",
        "nino3",
        "nino3_anom",
        "nino4",
        "nino4_anom",
        "nino34",
        "nino34_anom",
    ]

    df = pd.read_csv(
        url,
        sep=r"\s+",
        comment="#",
        names=col_names,
        header=None,
    )

    if df.empty:
        raise ValueError("NOAA 데이터 로딩 결과가 비어 있습니다.")

    df["date"] = pd.to_datetime(
        dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1)
    )
    return df.set_index("date").sort_index()


def load_oni_excel(path: str, sheet_name: str = "Sheet1") -> pd.Series:
    """Load ONI wide Excel format and convert to monthly-indexed series.

    Expected columns: year(또는 오탈자), DJF, JFM, ..., NDJ
    Each season value is mapped to its center month within the same year.
    """
    xls_path = Path(path)
    if not xls_path.exists():
        raise FileNotFoundError(f"ONI 엑셀 파일을 찾을 수 없습니다: {xls_path}")

    df = pd.read_excel(xls_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    if "year" in df.columns:
        year_col = "year"
    elif "Year" in df.columns:
        year_col = "Year"
    else:
        year_col = df.columns[0]

    records = []
    for _, row in df.iterrows():
        year_val = pd.to_numeric(row[year_col], errors="coerce")
        if pd.isna(year_val):
            continue
        year = int(year_val)

        for season, month in ONI_SEASON_TO_MONTH.items():
            if season not in df.columns:
                continue
            val = pd.to_numeric(row[season], errors="coerce")
            if pd.isna(val):
                continue
            records.append((pd.Timestamp(year=year, month=month, day=1), float(val)))

    if not records:
        raise ValueError("ONI 엑셀에서 유효한 시계열 레코드를 찾지 못했습니다.")

    series = pd.Series(
        data=[v for _, v in records],
        index=pd.DatetimeIndex([d for d, _ in records], name="date"),
        name="oni",
        dtype=np.float32,
    ).sort_index()

    series = series[~series.index.duplicated(keep="last")]
    return series


def split_and_scale(series: pd.Series, train_ratio: float, val_ratio: float) -> DatasetBundle:
    values = series.values.reshape(-1, 1).astype(np.float32)
    n = len(values)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = values[:train_end]
    val = values[train_end:val_end]
    test = values[val_end:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    return DatasetBundle(series=series, scaler=scaler, train_scaled=train_scaled, val_scaled=val_scaled, test_scaled=test_scaled)


def make_sequences(values: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(lookback, len(values)):
        x.append(values[i - lookback : i, 0])
        y.append(values[i, 0])

    x_arr = np.array(x, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    return np.expand_dims(x_arr, axis=-1), y_arr


def build_model(lookback: int, lstm_units: int, learning_rate: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(lookback, 1)),
            keras.layers.LSTM(lstm_units, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(max(8, lstm_units // 2)),
            keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def recursive_forecast(model: keras.Model, last_window_scaled: np.ndarray, steps: int) -> np.ndarray:
    window = last_window_scaled.copy().reshape(-1)
    preds = []
    for _ in range(steps):
        x = window.reshape(1, -1, 1)
        pred = model.predict(x, verbose=0)[0, 0]
        preds.append(pred)
        window = np.roll(window, -1)
        window[-1] = pred
    return np.array(preds, dtype=np.float32)


def plot_predictions(
    series: pd.Series,
    test_dates: pd.DatetimeIndex,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    future_dates: pd.DatetimeIndex,
    future_pred: np.ndarray,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, label="Observed", linewidth=1.2, alpha=0.7)
    plt.plot(test_dates, y_test, label="Test True", linewidth=2)
    plt.plot(test_dates, y_pred, label="Test Pred", linewidth=2)
    plt.plot(future_dates, future_pred, label="Future Forecast", linestyle="--", linewidth=2)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NOAA SST/ONI LSTM forecasting")
    parser.add_argument("--data-source", choices=["oni-excel", "noaa"], default="oni-excel")

    parser.add_argument("--oni-excel-path", default=DEFAULT_ONI_EXCEL_PATH, help="Path to ONI Excel file")
    parser.add_argument("--oni-sheet", default="Sheet1", help="Sheet name in ONI Excel")

    parser.add_argument("--url", default=NOAA_SST_INDEX_URL, help="NOAA SST index URL")
    parser.add_argument("--target", default="nino34", choices=["nino12", "nino3", "nino4", "nino34"])

    parser.add_argument("--lookback", type=int, default=24, help="Input window size in months")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--future-steps", type=int, default=12, help="Forecast horizon in months")
    parser.add_argument("--plot", default="forecast.png", help="Path to output plot file")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio 는 1.0 미만이어야 합니다.")

    keras.utils.set_random_seed(args.seed)

    if args.data_source == "oni-excel":
        series = load_oni_excel(args.oni_excel_path, sheet_name=args.oni_sheet)
        series_name = "ONI"
        y_label = "ONI (degC anomaly)"
        title = "ONI (Excel) - LSTM Forecast"
    else:
        df = load_noaa_sst_index(args.url)
        series = df[args.target].dropna().astype(np.float32)
        series_name = args.target
        y_label = "SST (degC)"
        title = f"NOAA SST ({args.target}) - LSTM Forecast"

    if len(series) <= args.lookback + 24:
        raise ValueError(
            f"데이터 길이가 부족합니다. len={len(series)}, lookback={args.lookback}. "
            "더 긴 시계열을 사용하거나 lookback을 줄이세요."
        )

    bundle = split_and_scale(series, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    if len(bundle.train_scaled) <= args.lookback:
        raise ValueError("학습 구간이 lookback보다 짧습니다. train_ratio 또는 lookback을 조정하세요.")

    val_context = np.vstack([bundle.train_scaled[-args.lookback :], bundle.val_scaled])
    test_context = np.vstack([val_context[-args.lookback :], bundle.test_scaled])

    x_train, y_train = make_sequences(bundle.train_scaled, args.lookback)
    x_val, y_val = make_sequences(val_context, args.lookback)
    x_test, y_test = make_sequences(test_context, args.lookback)

    model = build_model(args.lookback, args.lstm_units, args.lr)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    y_test_pred_scaled = model.predict(x_test, verbose=0).reshape(-1, 1)
    y_test_true = bundle.scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_test_pred = bundle.scaler.inverse_transform(y_test_pred_scaled).reshape(-1)

    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mae = mean_absolute_error(y_test_true, y_test_pred)

    print(f"Series:           {series_name}")
    print(f"Data points:      {len(series)}")
    print(f"Final train loss: {history.history['loss'][-1]:.6f}")
    print(f"Final val loss:   {history.history['val_loss'][-1]:.6f}")
    print(f"Test RMSE:        {rmse:.4f}")
    print(f"Test MAE:         {mae:.4f}")

    last_window = test_context[-args.lookback :]
    future_scaled = recursive_forecast(model, last_window_scaled=last_window, steps=args.future_steps)
    future_pred = bundle.scaler.inverse_transform(future_scaled.reshape(-1, 1)).reshape(-1)

    test_start = int(len(series) * (args.train_ratio + args.val_ratio))
    test_dates = series.index[test_start:]
    last_date = series.index[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=args.future_steps, freq="MS")

    plot_path = Path(args.plot)
    plot_predictions(
        series=series,
        test_dates=test_dates,
        y_test=y_test_true,
        y_pred=y_test_pred,
        future_dates=future_dates,
        future_pred=future_pred,
        output_path=plot_path,
        title=title,
        y_label=y_label,
    )

    print(f"Plot saved to: {plot_path.resolve()}")
    print("Future forecast:")
    for d, val in zip(future_dates, future_pred):
        print(f"  {d:%Y-%m}: {val:.3f}")


if __name__ == "__main__":
    main()
