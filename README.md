# ECNL-RL U14 Girls GLA Predictor — prototype

## What's included
- Match predictor with expected goals and home/draw/away probabilities
- Basic standings from imported scored Final rows
- Team home/away splits and recent results
- Results/data QA view and CSV export
- Explicit warnings that the current ledger is incomplete and the baseline model is experimental

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `python -m pip install -r requirements.txt`
4. Run `streamlit run app.py`

## Data caveats
This is a prototype, not a validated forecasting model. It uses the current workbook's `GLA Team Sources` list and `Verified Results` ledger as a starting point. A row marked Final is not necessarily fully reconciled. The app excludes rows without numeric scores and shows an explicit warning about limited coverage.

## Next development steps
1. Reconcile every result against source pages.
2. Add fixture import and result-entry validation.
3. Add team-specific and conference-aware ratings (goal-difference Elo + attack/defense Poisson).
4. Backtest chronological predictions and report score MAE plus probability Brier/log loss.
5. Deploy to a hosted service and connect a persistent database.
