"""Conservative refresh for ECNL-RL U14 Girls GLA.

Safeguards:
- Normalizes East/West to GLA East/GLA West.
- Identifies a fixture by date + the two teams (home/away independent).
- Never adds a fixture already in results.csv.
- Quarantines conflicting scores instead of overwriting saved results.
- Rejects future dates, implausible scores, unknown teams, and cross-conference
  candidates scraped from a conference page.
"""
from __future__ import annotations
import re, sys
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
RESULTS = ROOT / "results.csv"
TEAMS = ROOT / "teams.csv"
QUARANTINE = ROOT / "scrape_quarantine.csv"
SOURCES = [
    ("GLA East", "https://leaguetables.soccer/ecnl-rl-girls/u14/gla-east"),
    ("GLA West", "https://leaguetables.soccer/ecnl-rl-girls/u14/gla-west"),
]
ALIASES = {
    "Elite FC": "Elite FC ECNL RL G2012/13",
    "Steel City FC": "Steel City FC ECNL RL G2012/13",
    "Cincinnati Elite FC": "Cincinnati Elite FC ECNL RL G2012/13",
    "Columbus Force": "Columbus Force ECNL RL G2012/13",
    "FC Dayton": "FC Dayton ECNL RL G2012/13",
    "T3FC": "T3FC ECNL RL G2012/13",
    "Cuyahoga Valley Soccer Academy": "Cuyahoga Valley Soccer Academy ECNL RL G2012/13",
    "CVSA": "Cuyahoga Valley Soccer Academy ECNL RL G2012/13",
    "Northern Steel": "Northern Steel ECNL RL G2012/13",
    "Manta": "Manta ECNL RL G2012/13",
    "Canton Force": "Canton Force ECNL RL G2012/13",
    "WNY Flash Erie": "WNY Flash Erie G2012/13",
    "WNY Flash Rochester": "WNY Flash Rochester G2012/13",
}
ALIASES = dict(sorted(ALIASES.items(), key=lambda x: len(x[0]), reverse=True))
SCORE_RE = re.compile(r"^(\d{1,2})\s*[-ââ:]\s*(\d{1,2})$")
DATE_RE = re.compile(r"\b([A-Z][a-z]{2,8})\s+(\d{1,2}),\s+(20\d{2})\b")
COLS = ["date", "conference", "home_team", "away_team", "home_goals", "away_goals", "status", "source_url", "validation_note"]

def normalize_conf(value: str) -> str:
    value = str(value).strip().casefold().replace("gla ", "")
    if value == "east": return "GLA East"
    if value == "west": return "GLA West"
    if value == "cross-conference": return "Cross-conference"
    return str(value).strip()

def canon(value: str):
    value = " ".join(str(value).split()).casefold()
    for alias, canonical in ALIASES.items():
        if value == alias.casefold() or value == canonical.casefold():
            return canonical
    return None

def fetch(url: str) -> str:
    response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 (compatible; GLA-U14G-results-refresh/3.0)"})
    response.raise_for_status()
    if len(response.text) < 5000:
        raise RuntimeError(f"Unexpectedly short source page: {url}")
    return response.text

def parse(html: str, conference: str, url: str):
    soup = BeautifulSoup(html, "html.parser")
    tokens = [" ".join(s.split()) for s in soup.stripped_strings]
    found = []
    for i, token in enumerate(tokens):
        match = SCORE_RE.fullmatch(token.strip())
        if not match: continue
        before = [(j, canon(tokens[j])) for j in range(max(0, i-8), i) if canon(tokens[j])]
        after = [(j, canon(tokens[j])) for j in range(i+1, min(len(tokens), i+9)) if canon(tokens[j])]
        if not before or not after: continue
        _, home = max(before, key=lambda pair: pair[0])
        _, away = min(after, key=lambda pair: pair[0])
        if not home or not away or home == away: continue
        result_date = None
        for j in range(i, max(-1, i-35), -1):
            date_match = DATE_RE.search(tokens[j])
            if not date_match: continue
            month, day, year = date_match.groups()
            for fmt in ("%B %d %Y", "%b %d %Y"):
                try:
                    result_date = datetime.strptime(f"{month} {day} {year}", fmt).date().isoformat()
                    break
                except ValueError:
                    pass
            if result_date: break
        if not result_date: continue
        found.append({
            "date": result_date, "conference": normalize_conf(conference),
            "home_team": home, "away_team": away,
            "home_goals": int(match.group(1)), "away_goals": int(match.group(2)),
            "status": "Final â conservative auto-import", "source_url": url,
            "validation_note": "Auto-import candidate; verify if disputed.",
        })
    return found

def fixture_key(row):
    return (str(row["date"]), frozenset((str(row["home_team"]), str(row["away_team"]))))

def team_score_signature(row):
    return {str(row["home_team"]): int(row["home_goals"]), str(row["away_team"]): int(row["away_goals"])}

def main():
    if not RESULTS.exists() or not TEAMS.exists():
        raise RuntimeError("results.csv or teams.csv is missing")
    old = pd.read_csv(RESULTS, keep_default_na=False)
    if not set(COLS).issubset(old.columns):
        raise RuntimeError("results.csv does not have the expected columns; no changes made")
    old["conference"] = old["conference"].map(normalize_conf)
    team_df = pd.read_csv(TEAMS, keep_default_na=False)
    confmap = {str(row.team): normalize_conf(row.conference) for row in team_df.itertuples(index=False)}

    # Repair legacy short labels and remove duplicate fixture rows already in the ledger.
    # Keep the first occurrence (the canonical baseline was placed before scraper additions).
    cleaned_rows = []
    seen_existing = {}
    legacy_rejected = []
    for row in old.to_dict("records"):
        home_conf, away_conf = confmap.get(str(row["home_team"])), confmap.get(str(row["away_team"]))
        if home_conf and away_conf:
            row["conference"] = home_conf if home_conf == away_conf else "Cross-conference"
        key = fixture_key(row)
        signature = team_score_signature(row)
        if key in seen_existing:
            if signature == seen_existing[key]["signature"]:
                continue  # duplicate fixture with same result, often caused by East/West label mismatch
            legacy_rejected.append({**row, "quarantine_reason": "Conflicting duplicate already present in results.csv; first saved result retained"})
            continue
        seen_existing[key] = {"signature": signature}
        cleaned_rows.append(row)
    old = pd.DataFrame(cleaned_rows, columns=COLS)
    imported = []
    for conference, url in SOURCES:
        rows = parse(fetch(url), conference, url)
        print(f"{conference}: parsed {len(rows)} candidate score cards")
        imported.extend(rows)

    today = date.today()
    season_start = date(2026, 7, 1)
    accepted, rejected = [], []
    existing_by_key = {}
    for _, row in old.iterrows():
        existing_by_key.setdefault(fixture_key(row), []).append(row.to_dict())

    candidates = {}
    for row in imported:
        reason = ""
        try: game_date = date.fromisoformat(row["date"])
        except Exception: game_date = None; reason = "Invalid date"
        if not reason and not (season_start <= game_date <= today): reason = "Outside current season or future-dated"
        if not reason and (row["home_goals"] > 15 or row["away_goals"] > 15): reason = "Implausible score; possible parsing error"
        home_conf, away_conf = confmap.get(row["home_team"]), confmap.get(row["away_team"])
        if not reason and (not home_conf or not away_conf): reason = "Team not in configured 12-team list"
        if not reason and (home_conf != away_conf or home_conf != row["conference"]): reason = "Conference/source mismatch; manual review required"
        key = fixture_key(row)
        if not reason and key in existing_by_key:
            previous_rows = existing_by_key[key]
            candidate_sig = team_score_signature(row)
            prior_sigs = [team_score_signature(prior) for prior in previous_rows]
            if any(sig == candidate_sig for sig in prior_sigs):
                # Already present with the same score, even if conference labels differ.
                continue
            reason = "Existing fixture has a conflicting score; saved result preserved for manual review"
        if reason:
            rejected.append({**row, "quarantine_reason": reason})
        else:
            candidates.setdefault(key, []).append(row)

    for key, rows in candidates.items():
        signatures = {tuple(sorted(team_score_signature(row).items())) for row in rows}
        if len(signatures) > 1:
            rejected.extend({**row, "quarantine_reason": "Conflicting scraped candidates for same fixture"} for row in rows)
        else:
            accepted.append(rows[0])

    rejected = legacy_rejected + rejected
    if rejected:
        quarantine = pd.DataFrame(rejected)
        if QUARANTINE.exists():
            prior = pd.read_csv(QUARANTINE, keep_default_na=False)
            quarantine = pd.concat([prior, quarantine], ignore_index=True)
            dedupe_cols = [c for c in ["date", "home_team", "away_team", "home_goals", "away_goals", "quarantine_reason"] if c in quarantine.columns]
            quarantine = quarantine.drop_duplicates(subset=dedupe_cols, keep="first")
        quarantine.to_csv(QUARANTINE, index=False)
        print(f"Quarantined {len(rejected)} suspicious/conflicting candidates in scrape_quarantine.csv")
    out = pd.concat([old, pd.DataFrame(accepted, columns=COLS)], ignore_index=True)
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    out.to_csv(RESULTS, index=False)
    print(f"Added {len(accepted)} validated new results; cleaned/preserved {len(old)} existing unique fixtures.")
    print("Conference labels normalized. Existing fixtures are never duplicated or overwritten.")
    print("Review scrape_quarantine.csv for rejected candidates.")

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

