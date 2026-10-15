from __future__ import annotations

from pathlib import Path
from math import exp, factorial
import re
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = APP_DIR / "results.csv"
TEAMS_PATH = APP_DIR / "teams.csv"

st.set_page_config(
    page_title="ECNL-RL U14 Girls | GLA Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Visual system ----------
st.markdown("""
<style>
:root { --accent:#b8f36b; --panel:#171d27; --panel2:#202938; --muted:#aab4c4; }
.block-container {padding-top:1.1rem;padding-bottom:2.4rem;max-width:1280px;}
[data-testid="stHeader"] {background:transparent;}
.hero {padding:1.25rem 1.35rem;border:1px solid rgba(184,243,107,.24);border-radius:20px;
background:linear-gradient(120deg,#17212b 0%,#202c3b 58%,#18241e 100%);margin-bottom:1rem;}
.hero-kicker {color:var(--accent);font-size:.76rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;}
.hero h1 {font-size:clamp(1.55rem,4vw,2.45rem);line-height:1.15;margin:.35rem 0 .45rem;color:#f5f7fb;}
.hero p {margin:0;color:#c4cedb;font-size:.98rem;}
.metric-card {background:var(--panel);border:1px solid #303a49;border-radius:15px;padding:1rem 1.05rem;height:100%;}
.metric-label {color:var(--muted);font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
.metric-value {font-size:1.8rem;font-weight:800;line-height:1.25;margin-top:.25rem;color:#f5f7fb;}
.metric-note {font-size:.8rem;color:var(--muted);margin-top:.2rem;}
.section-title {font-size:1.18rem;font-weight:800;margin:.4rem 0 .7rem;color:#f5f7fb;}
.small-note {font-size:.84rem;color:var(--muted);}
.pred-card {background:linear-gradient(145deg,#1a2430,#131a22);border:1px solid #354151;border-radius:18px;padding:1rem;}
.pred-score {font-size:clamp(2rem,6vw,3.3rem);font-weight:900;letter-spacing:.03em;text-align:center;line-height:1.15;}
.pred-team {font-size:.94rem;font-weight:750;text-align:center;overflow-wrap:anywhere;}
.prob-wrap {background:#10151c;border-radius:999px;height:14px;overflow:hidden;display:flex;margin:.45rem 0 .3rem;}
.prob-home {background:#b8f36b;height:100%;}.prob-draw {background:#8592a6;height:100%;}.prob-away {background:#5ea8ff;height:100%;}
div[data-testid="stMetric"] {background:var(--panel);border:1px solid #303a49;padding:.8rem 1rem;border-radius:14px;}
div[data-testid="stMetricLabel"] p {color:var(--muted)!important;}
.stTabs [data-baseweb="tab-list"] {gap:.3rem;}
.stTabs [data-baseweb="tab"] {height:auto;padding:.7rem .9rem;border-radius:10px 10px 0 0;}
.stButton>button {border-radius:10px;font-weight:700;}
@media (max-width:640px) {
 .block-container {padding-left:.75rem;padding-right:.75rem;padding-top:.65rem;}
 .hero {padding:1rem;border-radius:15px;}
 .metric-card {padding:.8rem;}
 .metric-value {font-size:1.45rem;}
 .stTabs [data-baseweb="tab"] {padding:.55rem .58rem;font-size:.85rem;}
}
</style>
""", unsafe_allow_html=True)

# ---------- Load and normalize source data ----------
def load_teams() -> pd.DataFrame:
    if not TEAMS_PATH.exists():
        return pd.DataFrame(columns=["team", "conference"])
    df = pd.read_csv(TEAMS_PATH).fillna("")
    if not {"team", "conference"}.issubset(df.columns):
        return pd.DataFrame(columns=["team", "conference"])
    df["team"] = df["team"].astype(str).str.strip()
    df["conference"] = df["conference"].astype(str).str.strip().map(
        lambda x: "GLA East" if x.casefold().replace("gla ", "") == "east" else
                  "GLA West" if x.casefold().replace("gla ", "") == "west" else x
    )
    return df.drop_duplicates("team").sort_values("team").reset_index(drop=True)


def load_results() -> pd.DataFrame:
    cols = ["date", "conference", "home_team", "away_team", "home_goals", "away_goals", "status"]
    if not RESULTS_PATH.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(RESULTS_PATH).fillna("")
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["home_goals", "away_goals"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["conference"] = df["conference"].astype(str).str.strip().map(
        lambda x: "GLA East" if x.casefold().replace("gla ", "") == "east" else
                  "GLA West" if x.casefold().replace("gla ", "") == "west" else
                  "Cross-conference" if x.casefold() in {"cross-conference", "cross conference"} else x
    )
    df["status"] = df["status"].astype(str)
    # Only rows with real scores and final-like status count as completed results.
    final_mask = df["status"].str.contains(r"final|completed", case=False, regex=True)
    scored_mask = df["home_goals"].notna() & df["away_goals"].notna()
    df = df[final_mask & scored_mask].copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    # De-duplicate exact fixtures only; do not silently reconcile conflicting scores here.
    df["fixture_key"] = df.apply(lambda r: (r["date"], frozenset([r["home_team"], r["away_team"]])), axis=1)
    df = df.drop_duplicates("fixture_key", keep="first").drop(columns="fixture_key")
    return df.sort_values("date", ascending=False).reset_index(drop=True)

teams = load_teams()
results = load_results()
team_names = teams["team"].tolist()
conf_map = dict(zip(teams["team"], teams["conference"]))

# ---------- Standings and team summaries ----------
def empty_record(team: str) -> dict:
    return {"Team": team, "P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}


def standings(matches: pd.DataFrame, conference_only: bool = False) -> pd.DataFrame:
    rows = {t: empty_record(t) for t in team_names}
    for _, m in matches.iterrows():
        h, a = m["home_team"], m["away_team"]
        if h not in rows or a not in rows:
            continue
        if conference_only and (conf_map.get(h) != conf_map.get(a) or conf_map.get(h) not in {"GLA East", "GLA West"}):
            continue
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        for t, gf, ga in [(h, hg, ag), (a, ag, hg)]:
            rows[t]["P"] += 1; rows[t]["GF"] += gf; rows[t]["GA"] += ga
        if hg > ag:
            rows[h]["W"] += 1; rows[h]["Pts"] += 3; rows[h]["L"] += 0
            rows[a]["L"] += 1
        elif hg < ag:
            rows[a]["W"] += 1; rows[a]["Pts"] += 3; rows[h]["L"] += 1
        else:
            rows[h]["D"] += 1; rows[a]["D"] += 1; rows[h]["Pts"] += 1; rows[a]["Pts"] += 1
    out = pd.DataFrame(rows.values())
    out["GD"] = out["GF"] - out["GA"]
    out["Conference"] = out["Team"].map(conf_map)
    return out.sort_values(["Pts", "GD", "GF", "W", "Team"], ascending=[False, False, False, False, True]).reset_index(drop=True)


def team_matches(team: str, source: pd.DataFrame | None = None) -> pd.DataFrame:
    source = results if source is None else source
    if source.empty:
        return source.copy()
    return source[(source["home_team"] == team) | (source["away_team"] == team)].sort_values("date")


def team_form(team: str, limit: int = 5) -> list[str]:
    games = team_matches(team).tail(limit)
    form = []
    for _, m in games.iterrows():
        is_home = m["home_team"] == team
        gf = int(m["home_goals"] if is_home else m["away_goals"])
        ga = int(m["away_goals"] if is_home else m["home_goals"])
        form.append("W" if gf > ga else "L" if gf < ga else "D")
    return form

# ---------- Lightweight, explainable Poisson model ----------
def poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * (lam ** k) / factorial(k)


def prediction(home: str, away: str) -> dict:
    # League scoring baseline and per-team attack/defence estimates are shrunk
    # toward the cohort average to reduce overreaction to small samples.
    if results.empty:
        return {"home_xg": 1.25, "away_xg": 1.05, "home_win": 0.39, "draw": 0.29, "away_win": 0.32,
                "home_scores": [], "away_scores": [], "games_home": 0, "games_away": 0}
    all_goals = float(results["home_goals"].sum() + results["away_goals"].sum())
    avg_per_team_game = max(0.25, all_goals / max(1, 2 * len(results)))
    prior_games = 3.0

    def rates(team: str) -> tuple[float, float, int]:
        played = team_matches(team)
        n = len(played)
        gf = ga = 0.0
        for _, m in played.iterrows():
            if m["home_team"] == team:
                gf += float(m["home_goals"]); ga += float(m["away_goals"])
            else:
                gf += float(m["away_goals"]); ga += float(m["home_goals"])
        attack = (gf + prior_games * avg_per_team_game) / (n + prior_games)
        concede = (ga + prior_games * avg_per_team_game) / (n + prior_games)
        return attack, concede, n

    h_att, h_def, h_n = rates(home)
    a_att, a_def, a_n = rates(away)
    # Modest home advantage; cap rates for stability with youth-soccer small samples.
    home_xg = float(np.clip(0.58 * h_att + 0.42 * a_def + 0.18, 0.2, 4.2))
    away_xg = float(np.clip(0.58 * a_att + 0.42 * h_def, 0.2, 4.0))
    max_goals = 8
    matrix = np.array([[poisson_pmf(i, home_xg) * poisson_pmf(j, away_xg) for j in range(max_goals + 1)] for i in range(max_goals + 1)])
    # Normalize after truncation to make outcome probabilities sum to 100%.
    matrix = matrix / matrix.sum()
    home_win = float(np.tril(matrix, -1).sum())  # row score > column score
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    return {"home_xg": home_xg, "away_xg": away_xg, "home_win": home_win, "draw": draw, "away_win": away_win,
            "home_scores": h_att, "away_scores": a_att, "games_home": h_n, "games_away": a_n}

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">2026–27 Season · Great Lakes Alliance</div>
  <h1>⚽ ECNL-RL U14 Girls Predictor</h1>
  <p>Match forecasts, conference standings, and team performance — in one place.</p>
</div>
""", unsafe_allow_html=True)

completed = len(results)
latest_date = results["date"].max().strftime("%b %d, %Y") if completed and pd.notna(results["date"].max()) else "No results yet"
east_count = int((teams["conference"] == "GLA East").sum()) if not teams.empty else 0
west_count = int((teams["conference"] == "GLA West").sum()) if not teams.empty else 0
m1, m2, m3 = st.columns(3)
with m1: st.metric("Teams tracked", len(team_names))
with m2: st.metric("Completed matches", completed)
with m3: st.metric("Latest result date", latest_date)
st.caption(f"Coverage: {east_count} GLA East teams · {west_count} GLA West teams. Predictions are estimates based on the current results sample, not guarantees.")

if not team_names:
    st.error("Couldn't load teams.csv. Make sure teams.csv is in the same GitHub folder as app.py.")
    st.stop()

predict_tab, standings_tab, analytics_tab, results_tab = st.tabs(["⚽ Match Predictor", "🏆 Standings", "📈 Team Analytics", "🗓️ Results & Data"])

# ---------- Match Predictor ----------
with predict_tab:
    st.markdown('<div class="section-title">Matchup builder</div>', unsafe_allow_html=True)
    default_home = team_names.index("Elite FC ECNL RL G2012/13") if "Elite FC ECNL RL G2012/13" in team_names else 0
    default_away = team_names.index("T3FC ECNL RL G2012/13") if "T3FC ECNL RL G2012/13" in team_names else min(1, len(team_names)-1)
    left, mid, right = st.columns([5, 1, 5])
    with left: home = st.selectbox("Home team", team_names, index=default_home, key="pred_home")
    with mid: st.markdown("<div style='text-align:center;padding-top:2rem;font-weight:900;color:#b8f36b;font-size:1.1rem'>VS</div>", unsafe_allow_html=True)
    with right: away = st.selectbox("Away team", team_names, index=default_away, key="pred_away")
    if home == away:
        st.warning("Choose two different teams to create a matchup.")
    else:
        p = prediction(home, away)
        total_games = p["games_home"] + p["games_away"]
        if total_games < 4: confidence = "Very limited data"
        elif total_games < 10: confidence = "Early estimate"
        else: confidence = "More data available"
        st.markdown('<div class="section-title">Forecast</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([4, 3, 4])
        with c1:
            st.markdown(f'<div class="pred-card"><div class="pred-team">{home}</div><div class="pred-score">{p["home_xg"]:.1f}</div><div style="text-align:center;color:#aab4c4">Expected goals</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="pred-card"><div class="pred-team">Projected score</div><div class="pred-score">{round(p["home_xg"])}–{round(p["away_xg"])}</div><div style="text-align:center;color:#aab4c4">Model estimate</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="pred-card"><div class="pred-team">{away}</div><div class="pred-score">{p["away_xg"]:.1f}</div><div style="text-align:center;color:#aab4c4">Expected goals</div></div>', unsafe_allow_html=True)
        st.markdown("#### Result probabilities")
        probs = [("Home win", p["home_win"], "#b8f36b"), ("Draw", p["draw"], "#8592a6"), ("Away win", p["away_win"], "#5ea8ff")]
        cols = st.columns(3)
        for col, (label, val, color) in zip(cols, probs):
            with col: st.metric(label, f"{val:.0%}")
        st.markdown(f'''<div class="prob-wrap"><div class="prob-home" style="width:{p['home_win']*100:.3f}%"></div><div class="prob-draw" style="width:{p['draw']*100:.3f}%"></div><div class="prob-away" style="width:{p['away_win']*100:.3f}%"></div></div>''', unsafe_allow_html=True)
        st.caption("Home win · Draw · Away win, left to right")
        st.info(f"**How to read this:** The model blends each team's goals scored and conceded with the league-wide scoring average, shrinks small samples toward that average, and applies a modest home advantage. Current sample: {p['games_home']} matches for the home team and {p['games_away']} for the away team. **Data confidence: {confidence}.** Probabilities are model estimates, not betting odds or guarantees.")
        recent1, recent2 = st.columns(2)
        with recent1:
            st.markdown(f"**{home} · recent form**")
            form = team_form(home)
            st.write("  ".join(form) if form else "No completed matches yet")
        with recent2:
            st.markdown(f"**{away} · recent form**")
            form = team_form(away)
            st.write("  ".join(form) if form else "No completed matches yet")
        st.caption("Form is shown oldest to newest among the latest five completed matches. The model currently uses all available completed results; it does not yet apply recency weighting or head-to-head weighting.")

# ---------- Standings ----------
with standings_tab:
    st.markdown('<div class="section-title">League tables</div>', unsafe_allow_html=True)
    view = st.radio("Standings view", ["GLA West", "GLA East", "Overall"], horizontal=True, key="standings_view")
    conf_only = view != "Overall"
    table = standings(results, conference_only=conf_only)
    if view != "Overall": table = table[table["Conference"] == view].copy()
    display = table.copy()
    display.insert(0, "Rank", range(1, len(display) + 1))
    display = display.rename(columns={"Team":"Team", "P":"P", "W":"W", "D":"D", "L":"L", "GF":"GF", "GA":"GA", "GD":"GD", "Pts":"Pts"})
    display = display[["Rank", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]
    st.dataframe(display, use_container_width=True, hide_index=True, column_config={"Rank":st.column_config.NumberColumn("#", width="small"), "Team":st.column_config.TextColumn("Team", width="large"), "Pts":st.column_config.NumberColumn("Pts", help="3 points for a win, 1 for a draw")})
    st.caption("Conference tables include only matches between teams assigned to the same GLA conference. Cross-conference games count in overall standings only.")

# ---------- Team analytics ----------
with analytics_tab:
    st.markdown('<div class="section-title">Team profile</div>', unsafe_allow_html=True)
    selected = st.selectbox("Choose a team", team_names, key="analytics_team")
    games = team_matches(selected)
    overall = standings(results, conference_only=False).set_index("Team").loc[selected]
    conf = standings(results, conference_only=True).set_index("Team").loc[selected]
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Overall points", int(overall["Pts"]))
    with k2: st.metric("Goal difference", f"{int(overall['GD']):+d}")
    with k3: st.metric("Goals scored", int(overall["GF"]))
    with k4: st.metric("Goals conceded", int(overall["GA"]))
    st.markdown("#### Performance split")
    split = pd.DataFrame([
        {"Competition":"Overall", "Played":int(overall["P"]), "W":int(overall["W"]), "D":int(overall["D"]), "L":int(overall["L"]), "GF":int(overall["GF"]), "GA":int(overall["GA"]), "Pts":int(overall["Pts"])},
        {"Competition":"Conference", "Played":int(conf["P"]), "W":int(conf["W"]), "D":int(conf["D"]), "L":int(conf["L"]), "GF":int(conf["GF"]), "GA":int(conf["GA"]), "Pts":int(conf["Pts"])},
    ])
    st.dataframe(split, use_container_width=True, hide_index=True)
    if not games.empty:
        chart_rows = []
        for _, m in games.iterrows():
            is_home = m["home_team"] == selected
            chart_rows.append({"Date":m["date"], "Goals for":int(m["home_goals"] if is_home else m["away_goals"]), "Goals against":int(m["away_goals"] if is_home else m["home_goals"])})
        chart = pd.DataFrame(chart_rows).sort_values("Date").set_index("Date")
        st.markdown("#### Goals by match")
        st.bar_chart(chart, use_container_width=True)
        st.markdown("#### Recent results")
        out = games.copy()
        out["Date"] = out["date"].dt.strftime("%b %d, %Y")
        out["Score"] = out.apply(lambda r: f"{int(r['home_goals'])}–{int(r['away_goals'])}", axis=1)
        out["Venue"] = out.apply(lambda r: "Home" if r["home_team"] == selected else "Away", axis=1)
        out["Opponent"] = out.apply(lambda r: r["away_team"] if r["home_team"] == selected else r["home_team"], axis=1)
        st.dataframe(out[["Date", "Venue", "Opponent", "Score", "conference"]].rename(columns={"conference":"Match type"}), use_container_width=True, hide_index=True)
    else:
        st.info("No completed matches for this team are in the current results ledger.")

# ---------- Results & data ----------
with results_tab:
    st.markdown('<div class="section-title">Completed results</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1: team_filter = st.selectbox("Filter by team", ["All teams"] + team_names, key="results_team_filter")
    with f2: conf_filter = st.selectbox("Filter by match type", ["All", "GLA East", "GLA West", "Cross-conference"], key="results_conf_filter")
    shown = results.copy()
    if team_filter != "All teams": shown = shown[(shown["home_team"] == team_filter) | (shown["away_team"] == team_filter)]
    if conf_filter != "All": shown = shown[shown["conference"] == conf_filter]
    if not shown.empty:
        shown["Date"] = shown["date"].dt.strftime("%b %d, %Y")
        shown["Score"] = shown.apply(lambda r: f"{int(r['home_goals'])}–{int(r['away_goals'])}", axis=1)
        shown["Match"] = shown["home_team"] + "  vs  " + shown["away_team"]
        st.dataframe(shown[["Date", "conference", "Match", "Score"]].rename(columns={"conference":"Match type"}), use_container_width=True, hide_index=True)
    else:
        st.info("No completed results match those filters.")
    st.markdown("#### Data notes")
    st.write(f"- **{len(results)} unique completed fixtures** loaded from `results.csv`.")
    st.write(f"- Conference standings use only games where both teams belong to the same conference.")
    st.write(f"- Overall standings include cross-conference matches.")
    st.write(f"- Results are loaded from the repository at app startup; upcoming fixtures are not treated as completed unless scores and a final status are present.")
    if not results.empty and "validation_note" in results.columns:
        with st.expander("Show result validation notes"):
            notes = results.copy()
            notes["Date"] = notes["date"].dt.strftime("%b %d, %Y")
            notes["Match"] = notes["home_team"] + " vs " + notes["away_team"]
            notes["Score"] = notes.apply(lambda r: f"{int(r['home_goals'])}–{int(r['away_goals'])}", axis=1)
            st.dataframe(notes[["Date", "Match", "Score", "validation_note"]], use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("ECNL-RL U14 Girls · Great Lakes Alliance · 2026–27 | Forecasts are estimates and become more reliable as verified match data grows.")
