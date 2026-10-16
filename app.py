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
SCHEDULE_PATH = APP_DIR / "upcoming_schedule.csv"

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
.stButton>button {border-radius:10px;font-weight:700;min-height:2.8rem;white-space:normal;}
[data-testid="stDataFrame"] {max-width:100%;}
@media (max-width:640px) {
 .block-container {padding-left:.75rem;padding-right:.75rem;padding-top:.65rem;}
 .hero {padding:1rem;border-radius:15px;}
 .metric-card {padding:.8rem;}
 .metric-value {font-size:1.45rem;}
 .stTabs [data-baseweb="tab-list"] {display:flex;overflow-x:auto;flex-wrap:nowrap;scrollbar-width:thin;}
 .stTabs [data-baseweb="tab"] {padding:.55rem .58rem;font-size:.8rem;flex:0 0 auto;}
 .pred-card {padding:.7rem .4rem;border-radius:13px;}
 .pred-team {font-size:.78rem;line-height:1.25;}
 .pred-score {font-size:1.8rem;}
 div[data-testid="stMetric"] {padding:.65rem .7rem;}
 div[data-testid="stMetricValue"] {font-size:1.25rem;}
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

def load_schedule() -> pd.DataFrame:
    cols = ["date", "conference", "home_team", "away_team", "kickoff", "venue", "source_url"]
    if not SCHEDULE_PATH.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(SCHEDULE_PATH).fillna("")
    for c in cols:
        if c not in df.columns: df[c] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df = df[df["date"].notna() & df["home_team"].ne("") & df["away_team"].ne("")]
    df = df[df["date"] >= pd.Timestamp.now().normalize()]
    df["fixture_key"] = df.apply(lambda r: (r["date"], frozenset([r["home_team"], r["away_team"]])), axis=1)
    return df.drop_duplicates("fixture_key").drop(columns="fixture_key").sort_values(["date", "kickoff"]).reset_index(drop=True)

teams = load_teams()
results = load_results()
schedule = load_schedule()
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

# ---------- Regularized, recency-weighted Poisson model ----------
def poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * (lam ** k) / factorial(k)


def prediction(home: str, away: str) -> dict:
    """Regularized small-sample model with recency weighting and Poisson outcomes."""
    matches = results.copy()
    if matches.empty:
        home_xg, away_xg = 1.35, 1.10
        matrix = np.array([[poisson_pmf(i, home_xg) * poisson_pmf(j, away_xg)
                            for j in range(10)] for i in range(10)], dtype=float)
        matrix /= matrix.sum()
        return {"home_xg":home_xg,"away_xg":away_xg,"home_win":float(np.tril(matrix,-1).sum()),
                "draw":float(np.trace(matrix)),"away_win":float(np.triu(matrix,1).sum()),
                "games_home":0,"games_away":0,"confidence":"No results yet"}

    matches = matches[matches["home_team"].isin(team_names) & matches["away_team"].isin(team_names)].copy()
    if matches.empty:
        matches = results.copy()
    latest = matches["date"].max()
    age_days = (latest - matches["date"]).dt.days.clip(lower=0)
    matches["recency_weight"] = np.power(0.5, age_days / 35.0)
    matches["match_weight"] = matches["recency_weight"] * np.where(matches["conference"].eq("Cross-conference"), 0.85, 1.0)
    league_rate = max(0.35, float(matches["home_goals"].sum() + matches["away_goals"].sum()) / max(1.0, 2.0 * len(matches)))
    prior_games = 4.0

    def rates(team: str) -> tuple[float, float, int, float]:
        played = matches[(matches["home_team"] == team) | (matches["away_team"] == team)].copy()
        if played.empty:
            return league_rate, league_rate, 0, 0.0
        is_home = played["home_team"].eq(team)
        gf = np.where(is_home, played["home_goals"], played["away_goals"]).astype(float)
        ga = np.where(is_home, played["away_goals"], played["home_goals"]).astype(float)
        w = played["match_weight"].to_numpy(dtype=float)
        effective_n = float(w.sum())
        attack = (float(np.dot(gf, w)) + prior_games * league_rate) / (effective_n + prior_games)
        concede = (float(np.dot(ga, w)) + prior_games * league_rate) / (effective_n + prior_games)
        return attack, concede, len(played), effective_n

    h_att, h_concede, h_n, h_eff = rates(home)
    a_att, a_concede, a_n, a_eff = rates(away)
    home_xg = float(np.clip(0.52 * h_att + 0.48 * a_concede + 0.16, 0.20, 4.5))
    away_xg = float(np.clip(0.52 * a_att + 0.48 * h_concede, 0.20, 4.3))
    matrix = np.array([[poisson_pmf(i, home_xg) * poisson_pmf(j, away_xg)
                        for j in range(10)] for i in range(10)], dtype=float)
    matrix /= matrix.sum()
    effective_sample = h_eff + a_eff
    confidence = "Early estimate" if effective_sample < 8 else "Developing estimate" if effective_sample < 15 else "More data available"
    return {"home_xg":home_xg,"away_xg":away_xg,"home_win":float(np.tril(matrix,-1).sum()),
            "draw":float(np.trace(matrix)),"away_win":float(np.triu(matrix,1).sum()),
            "games_home":h_n,"games_away":a_n,"confidence":confidence}


# ---------- Matchup scouting report popup ----------
def team_profile_stats(team: str) -> dict:
    games = team_matches(team)
    if games.empty:
        return {"played": 0, "gf_pg": 0.0, "ga_pg": 0.0, "form": "No results", "home_record": "No home results", "away_record": "No away results"}

    gf = ga = wins = draws = losses = 0
    form = []
    for _, match in games.sort_values("date").iterrows():
        is_home = match["home_team"] == team
        scored = int(match["home_goals"] if is_home else match["away_goals"])
        conceded = int(match["away_goals"] if is_home else match["home_goals"])
        gf += scored
        ga += conceded
        result = "W" if scored > conceded else "D" if scored == conceded else "L"
        form.append(result)
        if result == "W": wins += 1
        elif result == "D": draws += 1
        else: losses += 1

    def split_record(is_home_value: bool) -> str:
        subset = games[games["home_team"] == team] if is_home_value else games[games["away_team"] == team]
        w = d = l = 0
        for _, match in subset.iterrows():
            scored = int(match["home_goals"] if is_home_value else match["away_goals"])
            conceded = int(match["away_goals"] if is_home_value else match["home_goals"])
            if scored > conceded: w += 1
            elif scored == conceded: d += 1
            else: l += 1
        return f"{w}-{d}-{l}" if len(subset) else "No games"

    return {
        "played": len(games), "gf_pg": gf / len(games), "ga_pg": ga / len(games),
        "form": "–".join(form[-5:]) if form else "No results",
        "home_record": split_record(True), "away_record": split_record(False),
        "wins": wins, "draws": draws, "losses": losses,
    }


@st.dialog("Matchup scouting report", width="large")
def show_matchup_report(home: str, away: str, forecast: dict) -> None:
    home_stats = team_profile_stats(home)
    away_stats = team_profile_stats(away)
    home_conf = conf_map.get(home, "Unknown conference")
    away_conf = conf_map.get(away, "Unknown conference")
    total_xg = forecast["home_xg"] + forecast["away_xg"]
    margin = forecast["home_xg"] - forecast["away_xg"]

    st.markdown("### 📍 Venue & context")
    st.write(f"**Listed home team:** {home}")
    st.write(f"**Matchup:** {home_conf} · {away_conf}")
    st.caption("The results ledger identifies home and away teams, but does not contain field addresses or venue names. The forecast treats the listed home team as the host and applies a modest home advantage; it cannot verify the actual field or travel conditions.")

    st.markdown("### ⚖️ How the teams compare")
    a, b = st.columns(2)
    with a:
        st.markdown(f"**{home}**")
        st.metric("Expected goals", f"{forecast['home_xg']:.1f}")
        st.write(f"Goals scored / match: **{home_stats['gf_pg']:.1f}**")
        st.write(f"Goals conceded / match: **{home_stats['ga_pg']:.1f}**")
        st.write(f"Last five form: **{home_stats['form']}**")
        st.write(f"Home record: **{home_stats['home_record']}**")
    with b:
        st.markdown(f"**{away}**")
        st.metric("Expected goals", f"{forecast['away_xg']:.1f}")
        st.write(f"Goals scored / match: **{away_stats['gf_pg']:.1f}**")
        st.write(f"Goals conceded / match: **{away_stats['ga_pg']:.1f}**")
        st.write(f"Last five form: **{away_stats['form']}**")
        st.write(f"Away record: **{away_stats['away_record']}**")

    st.markdown("### 🔭 What to expect")
    st.write(f"The model projects about **{total_xg:.1f} total goals** and an expected-goal difference of **{abs(margin):.1f}** in favor of **{home if margin >= 0 else away}**. The most likely rounded score is **{round(forecast['home_xg'])}–{round(forecast['away_xg'])}**; that score is a simple summary, not a certainty.")
    ph, pd_, pa = forecast["home_win"], forecast["draw"], forecast["away_win"]
    if max(ph, pd_, pa) < 0.45:
        st.info("This projects as a fairly open matchup: no single outcome dominates the model. The first goal and defensive mistakes could have a large effect.")
    elif ph > pa:
        st.info(f"The model leans toward the home team ({ph:.0%} home win probability), but a draw ({pd_:.0%}) or away win ({pa:.0%}) remains possible.")
    else:
        st.info(f"The model leans toward the away team ({pa:.0%} away win probability), despite the listed home advantage. A home win ({ph:.0%}) or draw ({pd_:.0%}) remains possible.")

    st.markdown("### 🏁 What each team needs to do to win")
    home_attack = home_stats["gf_pg"]
    away_defense = away_stats["ga_pg"]
    away_attack = away_stats["gf_pg"]
    home_defense = home_stats["ga_pg"]
    home_points = []
    away_points = []
    if home_attack >= away_defense:
        home_points.append("Use the attack to create sustained pressure; the historical scoring profile suggests opportunities may be available.")
    else:
        home_points.append("Be patient in possession and prioritize high-quality chances; the opponent's goals-against average suggests scoring may be harder.")
    if away_attack >= home_defense:
        away_points.append("Look to attack decisively and test the home defense; the scoring profile suggests the team can create chances.")
    else:
        away_points.append("Make transitions and set pieces count, and avoid giving away easy chances while building attacks carefully.")
    home_points.append("Protect against the opponent's main scoring threat and avoid turnovers in dangerous areas.")
    away_points.append("Manage the home team's early pressure, stay compact without the ball, and take chances when transitions open up.")
    if ph >= pa:
        home_points.append("The model's edge is on the home side: turn that into an early territorial advantage, but don't chase recklessly if the game stays level.")
        away_points.append("An away result likely requires disciplined defending and efficient finishing; keep the match close and capitalize on counterattacks.")
    else:
        home_points.append("The model does not give the home side the stronger forecast: defensive organization and converting limited chances are especially important.")
        away_points.append("The model favors the away side, but the team still needs to handle the venue disadvantage and avoid giving the home team momentum.")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{home}**")
        for point in home_points:
            st.markdown(f"- {point}")
    with right:
        st.markdown(f"**{away}**")
        for point in away_points:
            st.markdown(f"- {point}")

    st.caption(f"Model sample: {home_stats['played']} matches for the home team and {away_stats['played']} for the away team. This is a data-based scouting summary, not tactical video analysis. It does not know player availability, injuries, weather, actual field surface, or current lineups.")


# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">2026–27 Season · Great Lakes Alliance</div>
  <h1>⚽ ECNL-RL U14 Girls Predictor</h1>
  <p>Upcoming fixtures, simple match forecasts, and conference standings.</p>
</div>
""", unsafe_allow_html=True)

completed = len(results)
latest_date = results["date"].max().strftime("%b %d, %Y") if completed and pd.notna(results["date"].max()) else "No results yet"
east_count = int((teams["conference"] == "GLA East").sum()) if not teams.empty else 0
west_count = int((teams["conference"] == "GLA West").sum()) if not teams.empty else 0
m1, m2, m3 = st.columns(3)
with m1: st.metric("Teams", len(team_names))
with m2: st.metric("Upcoming fixtures", len(schedule))
with m3: st.metric("Completed matches", completed)
st.caption(f"Coverage: {east_count} GLA East teams · {west_count} GLA West teams. Predictions are estimates based on the current results sample, not guarantees.")

if not team_names:
    st.error("Couldn't load teams.csv. Make sure teams.csv is in the same GitHub folder as app.py.")
    st.stop()

schedule_tab, predict_tab, standings_tab, insights_tab = st.tabs(["🗓️ Schedule & Predictions", "⚽ Match Predictor", "🏆 Standings", "📈 Team Insights"])

# ---------- Upcoming schedule ----------
with schedule_tab:
    st.markdown('<div class="section-title">Upcoming GLA fixtures</div>', unsafe_allow_html=True)
    st.caption("Upcoming fixtures are separate from completed results and do not affect standings or model calculations.")
    if schedule.empty:
        st.info("No upcoming fixtures are loaded. Add confirmed fixtures to upcoming_schedule.csv in the app repository.")
    else:
        f1, f2 = st.columns(2)
        with f1: schedule_team = st.selectbox("Team", ["All teams"] + team_names, key="schedule_team")
        with f2: schedule_conf = st.selectbox("Conference", ["All conferences", "GLA East", "GLA West", "Cross-conference"], key="schedule_conf")
        shown_schedule = schedule.copy()
        if schedule_team != "All teams": shown_schedule = shown_schedule[(shown_schedule["home_team"] == schedule_team) | (shown_schedule["away_team"] == schedule_team)]
        if schedule_conf != "All conferences": shown_schedule = shown_schedule[shown_schedule["conference"] == schedule_conf]
        if shown_schedule.empty:
            st.info("No upcoming fixtures match those filters.")
        else:
            st.markdown("#### Upcoming games and projections")
            st.caption("Projected scores are estimates. The listed home team receives a small home advantage; verify kickoff and venue with the league or clubs.")
            for idx, (_, fixture) in enumerate(shown_schedule.iterrows()):
                forecast = prediction(fixture["home_team"], fixture["away_team"])
                date_label = fixture["date"].strftime("%a, %b %d")
                kickoff = str(fixture["kickoff"]).strip() or "Time TBD"
                venue = str(fixture["venue"]).strip() or "Venue TBD"
                with st.container(border=True):
                    st.markdown(f"**{date_label} · {kickoff}**")
                    st.markdown(f"<div style='text-align:center;font-size:.95rem;font-weight:750;line-height:1.4'>{fixture['home_team']}<br><span style='color:#b8f36b;font-size:1.65rem'>{round(forecast['home_xg'])} – {round(forecast['away_xg'])}</span><br>{fixture['away_team']}</div>", unsafe_allow_html=True)
                    st.caption(f"Expected goals: {forecast['home_xg']:.1f} – {forecast['away_xg']:.1f} · Home win {forecast['home_win']:.0%} · Draw {forecast['draw']:.0%} · Away win {forecast['away_win']:.0%}")
                    st.caption(f"Venue: {venue} · {forecast['confidence']}")
                    if st.button("Matchup analysis", use_container_width=True, key=f"schedule_matchup_{idx}"):
                        show_matchup_report(fixture["home_team"], fixture["away_team"], forecast)
        st.caption("Schedule source: LeagueTables.soccer. Verify kickoff and venue with the league or clubs.")

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
        confidence = p["confidence"]
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
        st.info(f"**How to read this:** The model blends each team's goals scored and conceded with the league-wide scoring average, shrinks small samples toward that average, weights recent results more heavily, slightly discounts cross-conference matches, and applies a modest home advantage. Current sample: {p['games_home']} matches for the home team and {p['games_away']} for the away team. **Data confidence: {confidence}.** Probabilities are model estimates, not betting odds or guarantees.")
        if st.button("🔎  Matchup analysis · venue, strengths & keys to win", use_container_width=True, type="primary", key="open_matchup_report"):
            show_matchup_report(home, away, p)
        st.caption("Open the matchup report for venue context, team comparisons, what to expect, and practical keys to winning for each side.")
        recent1, recent2 = st.columns(2)
        with recent1:
            st.markdown(f"**{home} · recent form**")
            form = team_form(home)
            st.write("  ".join(form) if form else "No completed matches yet")
        with recent2:
            st.markdown(f"**{away} · recent form**")
            form = team_form(away)
            st.write("  ".join(form) if form else "No completed matches yet")
        st.caption("Form is shown oldest to newest among the latest five completed matches. The model now weights recent results more heavily. It does not separately weight head-to-head results because the sample is too small to estimate that reliably.")

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

# ---------- Team insights ----------
with insights_tab:
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

    with st.expander("Completed results and data notes"):
        f1, f2 = st.columns(2)
        with f1: team_filter = st.selectbox("Filter by team", ["All teams"] + team_names, key="results_team_filter")
        with f2: conf_filter = st.selectbox("Filter by match type", ["All", "GLA East", "GLA West", "Cross-conference"], key="results_conf_filter")
        shown = results.copy()
        if team_filter != "All teams": shown = shown[(shown["home_team"] == team_filter) | (shown["away_team"] == team_filter)]
        if conf_filter != "All": shown = shown[shown["conference"] == conf_filter]
        if not shown.empty:
            shown["Date"] = shown["date"].dt.strftime("%b %d, %Y")
            shown["Score"] = shown.apply(lambda r: f"{int(r['home_goals'])}–{int(r['away_goals'])}", axis=1)
            shown["Match"] = shown["home_team"] + " vs " + shown["away_team"]
            st.dataframe(shown[["Date", "conference", "Match", "Score"]].rename(columns={"conference":"Match type"}), use_container_width=True, hide_index=True)
        else: st.info("No completed results match those filters.")
        st.write(f"- **{len(results)} unique completed fixtures** loaded from results.csv.")
        st.write("- Conference standings include same-conference games; overall standings include cross-conference matches.")
        st.write("- Upcoming fixtures are stored separately and never count as completed results.")

st.markdown("---")
st.caption("ECNL-RL U14 Girls · Great Lakes Alliance · 2026–27 | Forecasts are estimates and become more reliable as verified match data grows.")
