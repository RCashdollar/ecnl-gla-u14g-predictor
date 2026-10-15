import math
from pathlib import Path
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
teams_df = pd.read_csv(APP_DIR / "teams.csv")
if teams_df.empty or "team" not in teams_df.columns:
    st.error("teams.csv is empty or invalid. Restore the provided teams.csv file in the repository.")
    st.stop()
teams = sorted(teams_df["team"].dropna().astype(str).unique().tolist())
raw = pd.read_csv(APP_DIR / "results.csv", keep_default_na=False)

st.set_page_config(page_title="GLA U14 Girls Predictor", page_icon="⚽", layout="wide")
st.title("⚽ ECNL-RL U14 Girls — Great Lakes Alliance")
st.caption("2026–27 season • Experimental model • Results and predictions remain provisional until the source ledger is fully reconciled.")

source_url = "https://leaguetables.soccer/ecnl-rl-girls/u14"
st.info("Automatic results refresh is configured through the repository's daily GitHub Actions workflow. Imported scores are checked against LeagueTables' public U14 GLA East/West pages; if the source layout changes or a check fails, the last saved results are preserved.")
st.markdown(f"Results source: [LeagueTables — ECNL RL Girls U14]({source_url})")

with st.sidebar:
    st.header("Data controls")
    show_unverified = st.checkbox("Include non-final / unresolved rows in results view", value=False)
    st.caption("Rows marked Final (including user-confirmed finals) with numeric scores are used in the experimental baseline model.")
    st.markdown("---")
    st.markdown("**Scope:** 12 user-confirmed teams")
    st.markdown("**Current model:** experimental goal-rate baseline, not yet validated")

def final_rows(df):
    mask = df["status"].astype(str).str.lower().str.startswith("final")
    mask &= pd.to_numeric(df["home_goals"], errors="coerce").notna()
    mask &= pd.to_numeric(df["away_goals"], errors="coerce").notna()
    return df.loc[mask].copy()

def calculate_stats(matches):
    records = []
    for _, r in matches.iterrows():
        hg, ag = int(float(r["home_goals"])), int(float(r["away_goals"]))
        records.append((r["home_team"], hg, ag, True))
        records.append((r["away_team"], ag, hg, False))
    if not records:
        return pd.DataFrame(columns=["team","played","W","D","L","GF","GA","GD","points","home_played","away_played"])
    df = pd.DataFrame(records, columns=["team","gf","ga","is_home"])
    out=[]
    for team, g in df.groupby("team"):
        w=int((g.gf>g.ga).sum()); d=int((g.gf==g.ga).sum()); l=int((g.gf<g.ga).sum())
        out.append({
            "team":team, "played":len(g), "W":w, "D":d, "L":l,
            "GF":int(g.gf.sum()), "GA":int(g.ga.sum()), "GD":int(g.gf.sum()-g.ga.sum()),
            "points":3*w+d, "home_played":int(g.is_home.sum()), "away_played":int((~g.is_home).sum())
        })
    return pd.DataFrame(out)

def venue_stats(matches, team):
    played = matches[(matches.home_team == team) | (matches.away_team == team)].copy()
    out=[]
    for venue in ["Home","Away"]:
        g=played[played.home_team.eq(team)] if venue=="Home" else played[played.away_team.eq(team)]
        gf = pd.to_numeric(g.home_goals if venue=="Home" else g.away_goals, errors="coerce")
        ga = pd.to_numeric(g.away_goals if venue=="Home" else g.home_goals, errors="coerce")
        n=len(g)
        w=int((gf>ga).sum()); d=int((gf==ga).sum()); l=int((gf<ga).sum())
        out.append({"Venue":venue,"P":n,"W":w,"D":d,"L":l,
                    "GF/G":round(float(gf.sum()/n),2) if n else 0,
                    "GA/G":round(float(ga.sum()/n),2) if n else 0,
                    "GD/G":round(float((gf-ga).sum()/n),2) if n else 0})
    return pd.DataFrame(out)

def poisson_probs(lam):
    # P(X=k), k=0..9; tail probability is folded into 9 for a practical score matrix.
    vals=[math.exp(-lam)]
    for k in range(1,10):
        vals.append(vals[-1]*lam/k)
    vals[-1] += max(0.0, 1.0-sum(vals))
    return vals

def prediction(matches, home, away, home_adv):
    # Shrunk scoring baseline: league average plus team scoring/conceding tendencies.
    # This is intentionally conservative and marked experimental until sufficient validated data exists.
    if matches.empty:
        return None
    league_home = pd.to_numeric(matches.home_goals, errors="coerce").mean()
    league_away = pd.to_numeric(matches.away_goals, errors="coerce").mean()
    if pd.isna(league_home) or pd.isna(league_away):
        return None
    def team_rates(team):
        h=matches[matches.home_team.eq(team)]
        a=matches[matches.away_team.eq(team)]
        scored = pd.concat([
            pd.to_numeric(h.home_goals, errors="coerce"),
            pd.to_numeric(a.away_goals, errors="coerce")
        ]).dropna()
        conceded = pd.concat([
            pd.to_numeric(h.away_goals, errors="coerce"),
            pd.to_numeric(a.home_goals, errors="coerce")
        ]).dropna()
        # Prior of 3 matches at league scoring rate reduces wild small-sample swings.
        n=len(scored); prior=3
        attack=(scored.sum()+prior*((league_home+league_away)/2))/(n+prior) if n+prior else (league_home+league_away)/2
        ncon=len(conceded)
        defense=(conceded.sum()+prior*((league_home+league_away)/2))/(ncon+prior) if ncon+prior else (league_home+league_away)/2
        return attack, defense
    ha, hd = team_rates(home); aa, ad = team_rates(away)
    base=(league_home+league_away)/2
    # Multiplicative blend of attack and opponent concession, with conservative home bump.
    exp_home=max(0.15, ((ha/base) * (ad/base) * league_home) ** 0.5 + home_adv)
    exp_away=max(0.15, ((aa/base) * (hd/base) * league_away) ** 0.5)
    hp, ap=poisson_probs(exp_home), poisson_probs(exp_away)
    p_home=p_draw=p_away=0.0
    for i, pi in enumerate(hp):
        for j, pj in enumerate(ap):
            p=pi*pj
            if i>j: p_home+=p
            elif i==j: p_draw+=p
            else: p_away+=p
    best=max(((hp[i]*ap[j],i,j) for i in range(9) for j in range(9)), key=lambda x:x[0])
    return {"home_xg":exp_home,"away_xg":exp_away,"p_home":p_home,"p_draw":p_draw,"p_away":p_away,"score":(best[1],best[2])}

valid = final_rows(raw)
display_rows = raw if show_unverified else valid
tab1, tab2, tab3, tab4 = st.tabs(["Match Predictor", "Standings", "Team Analytics", "Results & Data QA"])

with tab1:
    st.subheader("Match Predictor")
    c1,c2,c3=st.columns([2,2,1])
    with c1:
        home=st.selectbox("Home team", teams, index=teams.index("Elite FC ECNL RL G2012/13") if "Elite FC ECNL RL G2012/13" in teams else 0)
    with c2:
        away_options=[t for t in teams if t != home]
        default_away=away_options.index("Steel City FC ECNL RL G2012/13") if "Steel City FC ECNL RL G2012/13" in away_options else 0
        away=st.selectbox("Away team", away_options, index=default_away)
    with c3:
        home_adv=st.slider("Home bump (goals)", 0.0, 0.6, 0.25, 0.05)
    st.caption(f"Model input: {len(valid)} rows currently marked Final with numeric scores. These still require source reconciliation before production use.")
    result=prediction(valid,home,away,home_adv)
    if result:
        a,b,c=st.columns(3)
        a.metric("Predicted score", f'{result["score"][0]} – {result["score"][1]}')
        b.metric("Expected goals", f'{result["home_xg"]:.2f} – {result["away_xg"]:.2f}')
        c.metric("Model status", "Experimental")
        p1,p2,p3=st.columns(3)
        p1.metric(f"{home} win", f'{result["p_home"]*100:.1f}%')
        p2.metric("Draw", f'{result["p_draw"]*100:.1f}%')
        p3.metric(f"{away} win", f'{result["p_away"]*100:.1f}%')
        st.warning("Prototype estimate only. Sparse, partly unreconciled results can materially distort these probabilities; do not treat them as betting-grade or validated forecasts.")
    else:
        st.info("Add at least one final match with a numeric score to enable the experimental prediction.")

with tab2:
    st.subheader("Provisional Conference Standings")
    if not valid.empty:
        overall_stats=calculate_stats(valid)
        overall_stats=overall_stats[["team","played","W","D","L"]].rename(columns={
            "played":"Overall P","W":"Overall W","D":"Overall D","L":"Overall L"})
        for conf_name in ["GLA East", "GLA West"]:
            conf_matches = valid[valid["conference"].eq(conf_name)]
            standings=calculate_stats(conf_matches)
            standings=standings[standings.team.isin(teams)]
            if not standings.empty:
                standings=standings.merge(overall_stats,on="team",how="left")
                standings=standings.sort_values(["points","GD","GF","team"], ascending=[False,False,False,True])
            st.markdown(f"**{conf_name}**")
            if standings.empty:
                st.info("No intra-conference results currently loaded.")
            else:
                standings=standings.rename(columns={"played":"Conf P","W":"Conf W","D":"Conf D","L":"Conf L",
                    "GF":"Conf GF","GA":"Conf GA","GD":"Conf GD","points":"Conf Pts"})
                st.dataframe(standings[["team","Overall P","Overall W","Overall L","Overall D","Conf P","Conf W","Conf L","Conf D","Conf GF","Conf GA","Conf GD","Conf Pts"]], use_container_width=True, hide_index=True)
        st.caption("Overall record includes all recorded league matches, including cross-conference games. Conference record counts only intra-conference games. W-L-T order is used throughout. Incomplete source histories mean these are provisional, not official standings.")
    else:
        st.info("No scored final results loaded yet.")

with tab3:
    st.subheader("Team Analytics")
    team=st.selectbox("Choose team", teams, key="analytics_team")
    stats=calculate_stats(valid)
    one=stats[stats.team.eq(team)]
    team_conf = teams_df.loc[teams_df["team"].eq(team), "conference"]
    team_conf = team_conf.iloc[0] if len(team_conf) else ""
    conf_matches = valid[valid["conference"].eq(team_conf)] if team_conf else valid.iloc[0:0]
    conf_stats=calculate_stats(conf_matches)
    conf_one=conf_stats[conf_stats.team.eq(team)]
    if not one.empty:
        r=one.iloc[0]
        cr=conf_one.iloc[0] if not conf_one.empty else None
        cols=st.columns(6)
        values=[("Overall record (W-L-T)",f'{r.W}-{r.L}-{r.D}'),
                ("Conference record (W-L-T)",f'{int(cr.W)}-{int(cr.L)}-{int(cr.D)}' if cr is not None else "0-0-0"),
                ("Overall played",r.played),("Overall goals",f'{r.GF}:{r.GA}'),
                ("Overall goal difference",f'{r.GD:+}'),("Overall points",r.points)]
        for col,(label,val) in zip(cols,values):
            col.metric(label,str(val))
    st.caption(f"Conference: {team_conf or 'not assigned'}. Overall includes cross-conference games; conference record includes only intra-conference games.")
    st.markdown("**Home vs away split (overall)**")
    st.dataframe(venue_stats(valid,team), use_container_width=True, hide_index=True)
    recent=valid[(valid.home_team.eq(team)) | (valid.away_team.eq(team))].sort_values("date",ascending=False).head(10)
    if not recent.empty:
        st.markdown("**Most recent imported results**")
        st.dataframe(recent[["date","home_team","home_goals","away_goals","away_team","source_url"]], use_container_width=True, hide_index=True)
    else:
        st.info("No validated results currently associated with this team.")

with tab4:
    st.subheader("Results & Data QA")
    st.markdown("Use this view to inspect what the app has imported. Unresolved rows are excluded from the model by default.")
    st.metric("Rows in current ledger", len(raw))
    st.metric("Final scored rows used by baseline", len(valid))
    view=display_rows.copy()
    if not view.empty:
        cols=[c for c in ["date","conference","home_team","home_goals","away_goals","away_team","status","validation_note","source_url"] if c in view.columns]
        st.dataframe(view[cols], use_container_width=True, hide_index=True)
    st.markdown("**Reconciliation status:** conference labels are normalized and cross-conference games are excluded from standings. Continue checking reciprocal result histories and fixture updates before fitting a production model.")
    st.download_button("Download current results CSV", raw.to_csv(index=False).encode("utf-8"), "gla_u14g_results.csv", "text/csv")

st.markdown("---")
st.caption("Results are stored in this repository and refreshed by a scheduled workflow. LeagueTables reports computed standings; check the league or club for official rulings and score corrections.")
