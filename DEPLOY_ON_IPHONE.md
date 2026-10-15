# Put the ECNL GLA prototype online

## Deploy with Streamlit Community Cloud
1. Sign in to GitHub and create a new repository named `ecnl-gla-u14g-predictor`.
2. Extract this ZIP, then upload the files and the `.streamlit` folder into the repository root. Make sure `app.py`, `requirements.txt`, `teams.csv`, and `results.csv` are at the top level.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app** → **Yup, I have an app**.
5. Choose repository `ecnl-gla-u14g-predictor`, branch `main`, and file path `app.py`.
6. Click **Deploy**. Wait for Streamlit to finish building, then open the app URL it provides.
7. On iPhone, open the app URL in Safari → Share → Add to Home Screen.

## Privacy
This prototype contains youth soccer team and match information, not personal player data. A public GitHub repository generally results in a public app. If you want access limited to you, check the current Streamlit Community Cloud private-app options before deploying.

## Current limitations
This is a prototype with a small, partly reconciled result ledger. Predictions are experimental, and match schedules/results are not live-synced.
