# Model card: trend + weekday OLS

**Owner:** Arwa Ahmed. **Status:** educational/portfolio baseline; no production validation.

## Intended use

Explain a reproducible daily-series forecasting workflow with explicit chronological data boundaries. Inputs are consecutive nonnegative daily demand observations. Outputs are point forecasts and a residual-based interval illustration. Not for autonomous purchasing, financial commitments or safety-critical forecasting.

## Training and data provenance

The default dataset is generated locally by `synthetic(days=196, seed=17)`: linear growth, additive weekday pattern and Gaussian noise. No personal data, customer dataset or external weights are included. With a user CSV, responsibility for rights, provenance and representativeness remains with the operator. The report records whether a synthetic generator or user CSV was used.

The model fits an intercept, weekly-scaled time trend and six weekday indicators on the training period only. Calibration residuals determine the displayed band radius without refitting. The test period affects metrics only. The serialized model records fitting and calibration cutoffs.

## Evidence

`examples/report.json` contains metrics from the bundled synthetic run, and `examples/predictions.csv` contains every held-out observation and prediction. These demonstrate reproducibility, not real-world efficacy. The generator's trend/weekday structure favors the hypothesis class. The seasonal-naive reference has access to more recent pre-test observations than the frozen fitted model.

## Limitations and failure modes

Nonlinear growth, changing weekday patterns, special events, intermittent demand, missing dates and regime changes are not modeled. Residual dependence and distribution drift invalidate exchangeability-based coverage interpretations, so bands are explicitly heuristic. Predictions are clipped at zero. The normal-equation solver is appropriate for the small demonstration but less numerically robust than QR/SVD for ill-conditioned large designs.

## Before real use

Validate business data rights and semantics, compare rolling-origin baselines, measure error by product/segment and operational cost, assess drift, document imputation, and review forecasts with the responsible domain owner. No hosting, monitoring, retraining scheduler or production approval is supplied by this project.
