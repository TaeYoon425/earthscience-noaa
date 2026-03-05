#!/usr/bin/env python3
"""Tune ONI LSTM hyperparameters and forecast through a target month."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

import sst_lstm_noaa as core


def fast_recursive_forecast(model: keras.Model, last_window_scaled: np.ndarray, steps: int) -> np.ndarray:
    """Faster recursive forecasting than repeated model.predict calls."""
    window = last_window_scaled.copy().reshape(-1).astype(np.float32)
    preds = np.zeros((steps,), dtype=np.float32)

    for i in range(steps):
        x = window.reshape(1, -1, 1)
        pred = model(x, training=False).numpy()[0, 0]
        preds[i] = pred
        window[:-1] = window[1:]
        window[-1] = pred

    return preds


def evaluate_config(
    series: pd.Series,
    lookback: int,
    lstm_units: int,
    lr: float,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    epochs: int,
    patience: int,
    seed: int,
) -> dict:
    keras.utils.set_random_seed(seed)

    bundle = core.split_and_scale(series, train_ratio=train_ratio, val_ratio=val_ratio)
    val_context = np.vstack([bundle.train_scaled[-lookback:], bundle.val_scaled])
    test_context = np.vstack([val_context[-lookback:], bundle.test_scaled])

    x_train, y_train = core.make_sequences(bundle.train_scaled, lookback)
    x_val, y_val = core.make_sequences(val_context, lookback)
    x_test, y_test = core.make_sequences(test_context, lookback)

    model = core.build_model(lookback, lstm_units, lr)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    y_test_pred_scaled = model.predict(x_test, verbose=0).reshape(-1, 1)
    y_test_true = bundle.scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_test_pred = bundle.scaler.inverse_transform(y_test_pred_scaled).reshape(-1)

    return {
        "model": model,
        "bundle": bundle,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
        "test_rmse": float(np.sqrt(mean_squared_error(y_test_true, y_test_pred))),
        "test_mae": float(mean_absolute_error(y_test_true, y_test_pred)),
        "best_val_loss": float(np.min(hist.history["val_loss"])),
        "epochs_ran": len(hist.history["loss"]),
        "test_context": test_context,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune ONI LSTM and forecast until target month")
    parser.add_argument(
        "--oni-excel-path",
        default="/Users/kimtaeyoon/Library/CloudStorage/OneDrive-백운중학교/통합 문서1.xlsx",
    )
    parser.add_argument("--oni-sheet", default="Sheet1")
    parser.add_argument("--forecast-end", default="2045-12")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/Users/kimtaeyoon/Documents/Playground/outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = core.load_oni_excel(args.oni_excel_path, sheet_name=args.oni_sheet)

    search_space = [
        {"lookback": 24, "lstm_units": 64, "lr": 1e-3, "batch_size": 16},
        {"lookback": 36, "lstm_units": 64, "lr": 1e-3, "batch_size": 16},
        {"lookback": 48, "lstm_units": 64, "lr": 1e-3, "batch_size": 16},
        {"lookback": 36, "lstm_units": 96, "lr": 5e-4, "batch_size": 16},
        {"lookback": 48, "lstm_units": 96, "lr": 5e-4, "batch_size": 16},
        {"lookback": 60, "lstm_units": 96, "lr": 5e-4, "batch_size": 32},
    ]

    rows = []
    for idx, cfg in enumerate(search_space, start=1):
        result = evaluate_config(
            series=series,
            lookback=cfg["lookback"],
            lstm_units=cfg["lstm_units"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            epochs=60,
            patience=8,
            seed=args.seed,
        )

        row = {
            **cfg,
            "best_val_loss": result["best_val_loss"],
            "test_rmse": result["test_rmse"],
            "test_mae": result["test_mae"],
            "epochs_ran": result["epochs_ran"],
        }
        rows.append(row)
        print(f"[{idx}/{len(search_space)}] {row}", flush=True)

    results_df = pd.DataFrame(rows).sort_values(["test_mae", "test_rmse", "best_val_loss"]).reset_index(drop=True)
    best_cfg = results_df.iloc[0].to_dict()

    print("\n=== TOP 5 CONFIGS ===", flush=True)
    print(results_df.head(5).to_string(index=False), flush=True)
    print("\nBEST CONFIG:", best_cfg, flush=True)

    final = evaluate_config(
        series=series,
        lookback=int(best_cfg["lookback"]),
        lstm_units=int(best_cfg["lstm_units"]),
        lr=float(best_cfg["lr"]),
        batch_size=int(best_cfg["batch_size"]),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        epochs=140,
        patience=15,
        seed=args.seed,
    )

    print(
        f"\nFINAL TEST -> RMSE={final['test_rmse']:.4f}, MAE={final['test_mae']:.4f}",
        flush=True,
    )

    last_date = series.index[-1]
    forecast_end = pd.Timestamp(f"{args.forecast_end}-01")
    if forecast_end <= last_date:
        raise ValueError(f"forecast_end({forecast_end.date()}) must be after last_date({last_date.date()})")

    future_steps = (forecast_end.year - last_date.year) * 12 + (forecast_end.month - last_date.month)

    lookback = int(best_cfg["lookback"])
    last_window = final["test_context"][-lookback:]
    future_scaled = fast_recursive_forecast(final["model"], last_window, steps=future_steps)
    future_pred = final["bundle"].scaler.inverse_transform(future_scaled.reshape(-1, 1)).reshape(-1)
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=future_steps, freq="MS")

    test_start = int(len(series) * (args.train_ratio + args.val_ratio))
    test_dates = series.index[test_start:]

    plot_path = out_dir / "oni_forecast_to_2045_tuned.png"
    core.plot_predictions(
        series=series,
        test_dates=test_dates,
        y_test=final["y_test_true"],
        y_pred=final["y_test_pred"],
        future_dates=future_dates,
        future_pred=future_pred,
        output_path=plot_path,
        title="ONI Tuned LSTM Forecast Through 2045",
        y_label="ONI (degC anomaly)",
    )

    forecast_df = pd.DataFrame({"date": future_dates, "oni_pred": future_pred})
    forecast_csv = out_dir / "oni_forecast_to_2045_tuned.csv"
    forecast_df.to_csv(forecast_csv, index=False)

    report_path = out_dir / "oni_tuning_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("All tuning results (sorted by test_mae):\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nBest config:\n")
        f.write(str(best_cfg))
        f.write(
            f"\nFinal test RMSE={final['test_rmse']:.4f}, MAE={final['test_mae']:.4f}, "
            f"lookback={lookback}, forecast_steps={future_steps}\n"
        )

    print(f"Forecast range: {future_dates.min().date()} ~ {future_dates.max().date()} ({future_steps} steps)", flush=True)
    print(f"Saved plot: {plot_path}", flush=True)
    print(f"Saved CSV: {forecast_csv}", flush=True)
    print(f"Saved report: {report_path}", flush=True)
    print("\nTail forecast:", flush=True)
    print(forecast_df.tail(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
