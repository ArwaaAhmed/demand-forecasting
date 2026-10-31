import unittest
from datetime import date, timedelta

from forecast import evaluate, fit, predict, seasonal_naive, synthetic, validate


class ForecastTests(unittest.TestCase):
    def test_recovers_known_linear_weekday_process(self):
        start = date(2025, 1, 1)
        rows = [{"date": (start + timedelta(days=i)).isoformat(), "demand": 50 + i * 0.25 + (i % 7) * 2} for i in range(100)]
        model = fit(rows[:70])
        output = predict(model, [r["date"] for r in rows[70:]])
        for expected, actual in zip(rows[70:], output):
            self.assertAlmostEqual(expected["demand"], actual, places=8)

    def test_test_targets_do_not_change_fit_or_predictions(self):
        rows = synthetic()
        original_report, original_model, original_predictions = evaluate(rows)
        changed = [dict(r) for r in rows]
        for row in changed[-28:]:
            row["demand"] += 500
        changed_report, changed_model, changed_predictions = evaluate(changed)
        self.assertEqual(original_model, changed_model)
        self.assertEqual([r["prediction"] for r in original_predictions], [r["prediction"] for r in changed_predictions])
        self.assertNotEqual(original_report["model"], changed_report["model"])

    def test_calibration_targets_do_not_refit_model(self):
        rows = synthetic()
        _, original, _ = evaluate(rows)
        for row in rows[-56:-28]:
            row["demand"] += 100
        _, changed, _ = evaluate(rows)
        self.assertEqual(original["coefficients"], changed["coefficients"])
        self.assertNotEqual(original["heuristic_interval_radius"], changed["heuristic_interval_radius"])

    def test_rejects_missing_dates_and_invalid_values(self):
        for rows in [[{"date": "2025-01-01", "demand": -1}], [{"date": "2025-01-01", "demand": float("nan")}],
                     [{"date": "2025-01-01", "demand": 1}, {"date": "2025-01-03", "demand": 2}]]:
            with self.assertRaises(ValueError):
                validate(rows)

    def test_seasonal_baseline_repeats_last_week(self):
        training = [{"demand": float(i)} for i in range(10)]
        self.assertEqual(seasonal_naive(training, 9), [3, 4, 5, 6, 7, 8, 9, 3, 4])


if __name__ == "__main__":
    unittest.main()
