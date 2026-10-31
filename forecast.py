"""Explainable daily-demand forecasting, using only the Python standard library."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from statistics import mean


def synthetic(days: int = 196, seed: int = 17) -> list[dict]:
    """Create an invented retail series; it is not a customer dataset."""
    if days < 1:
        raise ValueError("days must be positive")
    rng = random.Random(seed)
    start = date(2025, 1, 1)
    weekly = [-8, -5, -2, 1, 7, 15, 10]
    return [{"date": (start + timedelta(days=i)).isoformat(),
             "demand": round(max(0, 80 + 0.32 * i + weekly[(start + timedelta(days=i)).weekday()]
                                   + rng.gauss(0, 3.5)), 3)} for i in range(days)]


def validate(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("the series is empty")
    clean = []
    for row in rows:
        day, value = date.fromisoformat(row["date"]), float(row["demand"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("demand must be finite and non-negative")
        if clean and day != date.fromisoformat(clean[-1]["date"]) + timedelta(days=1):
            raise ValueError("dates must be unique, ordered and consecutive; impute deliberately upstream")
        clean.append({"date": day.isoformat(), "demand": value})
    return clean


def features(day: date, origin: date) -> list[float]:
    # Weeks keep the time feature numerically comparable to weekday indicators.
    return [1.0, (day - origin).days / 7] + [float(day.weekday() == i) for i in range(6)]


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Partial-pivot Gaussian elimination for the small full-rank design matrix."""
    a = [row[:] + [value] for row, value in zip(matrix, vector)]
    n = len(vector)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-10:
            raise ValueError("insufficient independent observations")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for row in range(n):
            if row == col:
                continue
            scale = a[row][col]
            a[row] = [left - scale * right for left, right in zip(a[row], a[col])]
    return [row[-1] for row in a]


def fit(rows: list[dict]) -> dict:
    rows = validate(rows)
    if len(rows) < 28:
        raise ValueError("at least 28 training days are required")
    origin = date.fromisoformat(rows[0]["date"])
    x = [features(date.fromisoformat(row["date"]), origin) for row in rows]
    y = [row["demand"] for row in rows]
    gram = [[sum(row[i] * row[j] for row in x) for j in range(8)] for i in range(8)]
    rhs = [sum(row[i] * target for row, target in zip(x, y)) for i in range(8)]
    return {"model": "linear-trend-weekday-ols", "origin": origin.isoformat(),
            "trained_through": rows[-1]["date"], "training_rows": len(rows),
            "coefficients": solve(gram, rhs)}


def predict(model: dict, days: list[str]) -> list[float]:
    origin = date.fromisoformat(model["origin"])
    return [max(0.0, sum(a * b for a, b in zip(model["coefficients"], features(date.fromisoformat(day), origin))))
            for day in days]


def seasonal_naive(training: list[dict], horizon: int) -> list[float]:
    if len(training) < 7 or horizon < 1:
        raise ValueError("seasonal naive requires seven training days and a positive horizon")
    # Repeat the final observed week; never read actual future target values.
    return [training[-7 + i % 7]["demand"] for i in range(horizon)]


def metrics(actual: list[float], predicted: list[float]) -> dict:
    if not actual or len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal, non-zero length")
    errors = [a - p for a, p in zip(actual, predicted)]
    return {"mae": mean(abs(e) for e in errors), "rmse": math.sqrt(mean(e * e for e in errors))}


def evaluate(rows: list[dict], calibration_days: int = 28, test_days: int = 28) -> tuple[dict, dict, list[dict]]:
    rows = validate(rows)
    if calibration_days < 7 or test_days < 7 or len(rows) - calibration_days - test_days < 28:
        raise ValueError("need 28+ training days and calibration/test windows of at least seven days")
    training = rows[: -calibration_days - test_days]
    calibration = rows[-calibration_days - test_days: -test_days]
    test = rows[-test_days:]
    model = fit(training)
    # Freeze the fit before calibration; neither calibration nor test refits it.
    residuals = sorted(abs(r["demand"] - p) for r, p in zip(calibration, predict(model, [r["date"] for r in calibration])))
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * 0.90))
    radius = residuals[rank - 1]
    model["calibrated_through"] = calibration[-1]["date"]
    model["heuristic_interval_radius"] = radius
    prediction = predict(model, [r["date"] for r in test])
    baseline = seasonal_naive(rows[:-test_days], test_days)
    results = [{**row, "prediction": p, "seasonal_naive": b, "lower": max(0, p - radius), "upper": p + radius}
               for row, p, b in zip(test, prediction, baseline)]
    actual = [r["demand"] for r in test]
    report = {"data_source": "caller-supplied; see run metadata", "split": {
        "training": [training[0]["date"], training[-1]["date"]],
        "calibration": [calibration[0]["date"], calibration[-1]["date"]],
        "test": [test[0]["date"], test[-1]["date"]]},
        "model": metrics(actual, prediction), "seasonal_naive": metrics(actual, baseline),
        "interval": {"nominal_level": 0.90, "observed_test_coverage": mean(r["lower"] <= r["demand"] <= r["upper"] for r in results),
                     "caveat": "Residual band is heuristic for temporal data; no guaranteed future coverage."},
        "evaluation_note": "Model is frozen before calibration. Baseline can use the last observed calibration week; test targets are hidden from both predictors."}
    return report, model, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="optional consecutive daily date,demand CSV")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--test-days", type=int, default=28)
    args = parser.parse_args()
    if args.csv:
        with args.csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        source = {"type": "user_csv", "filename": args.csv.name}
    else:
        rows = synthetic(seed=args.seed)
        source = {"type": "synthetic_demo", "seed": args.seed, "generator": "trend + weekday + Gaussian noise; no real customers"}
    report, model, predictions = evaluate(rows, test_days=args.test_days)
    report["data_source"] = source
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [("report.json", report), ("model.json", model)]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps({"report": str(args.output / "report.json"), "metrics": report["model"]}, indent=2))


if __name__ == "__main__":
    main()
