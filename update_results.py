"""Refresh GLA U14 Girls results from LeagueTables' public conference pages.

Safety behavior: never removes existing rows; only promotes a scheduled/unresolved
fixture to a final score when a recognizable team pair and numeric score are found.
If the source page layout changes or no scores can be parsed, exits non-zero and
leaves results.csv unchanged so GitHub Actions won't commit a broken update.
"""
from __future__ import annotations
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
RESULTS = ROOT / "results.csv"
TEAMS = ROOT / "teams.csv"
SOURCES = [
    ("GLA East", "https://leaguetables.soccer/ecnl-rl-girls/u14/gla-east"),
    ("GLA West", "https://leaguetables.soccer/ecnl-rl-girls/u14/gla-west"),
]

# Names as displayed by LeagueTables mapped to the app's canonical team labels.
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
# Longest aliases first to avoid matching "Elite FC" inside "Cincinnati Elite FC".
ALIASES = dict(sorted(ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True))
TEAM_PATTERN = re.compile("|".join(re.escape(x) for x in ALIASES), re.I)
SCORE_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*[-–—:]\s*(\d{1,2})(?!\d)")
DATE_PATTERN = re.compile(r"\b([A-Z][a-z]{2,8})\s+(\d{1,2}),\s+(20\d{2})\b")


def canonical_team(text: str):
    for alias, canonical in ALIASES.items():
        if text.casefold() == alias.casefold():
            return canonical
    return None


def fetch_page(url: str) -> str:
    response = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (compatible; GLA-U14G-results-refresh/1.0; respectful daily check)"
    })
    response.raise_for_status()
    if len(response.text) < 5000:
        raise RuntimeError(f"Unexpectedly short response from {url}: {len(response.text)} bytes")
    return response.text


def parse_page(html: str, conference: str, url: str):
    soup = BeautifulSoup(html, "html.parser")
    # Keep individual DOM text nodes so home/away names remain near their score.
    tokens = [" ".join(s.split()) for s in soup.stripped_strings]
    found = []
    for i, token in enumerate(tokens):
        score = SCORE_PATTERN.fullmatch(token.strip())
        if not score:
            # Some pages wrap score text with adjacent whitespace or status labels.
            m = SCORE_PATTERN.search(token)
            if not m or len(token.strip()) > 14:
                continue
            score = m
        left_team = right_team = None
        left_i = right_i = None
        for j in range(i - 1, max(-1, i - 18), -1):
            candidate = canonical_team(tokens[j].strip())
            if candidate:
                left_team, left_i = candidate, j
                break
        for j in range(i + 1, min(len(tokens), i + 19)):
            candidate = canonical_team(tokens[j].strip())
            if candidate:
                right_team, right_i = candidate, j
                break
        if not left_team or not right_team or left_team == right_team:
            continue
        # Date is typically a heading before the match cards; use nearest preceding date.
        date_value = None
        for j in range(i, max(-1, i - 80), -1):
            dm = DATE_PATTERN.search(tokens[j])
            if dm:
                try:
                    date_value = datetime.strptime(" ".join(dm.groups()), "%B %d %Y").date().isoformat()
                except ValueError:
                    try:
                        date_value = datetime.strptime(" ".join(dm.groups()), "%b %d %Y").date().isoformat()
                    except ValueError:
                        pass
                if date_value:
                    break
        if not date_value:
            continue
        # Match cards normally list home then away around the score. Avoid cross-card
        # matches by requiring both team tokens within a small window.
        if i - left_i > 10 or right_i - i > 10:
            continue
        found.append({
            "date": date_value, "conference": conference,
            "home_team": left_team, "away_team": right_team,
            "home_goals": int(score.group(1)), "away_goals": int(score.group(2)),
            "status": "Final — LeagueTables auto-import",
            "source_url": url,
            "validation_note": "Automatically imported from public LeagueTables result card; verify source corrections if disputed.",
        })
    # Deduplicate repeated snippets/cards.
    unique = {}
    for row in found:
        key = (row["date"], row["home_team"], row["away_team"], row["home_goals"], row["away_goals"])
        unique[key] = row
    return list(unique.values())


def main():
    existing = pd.read_csv(RESULTS, keep_default_na=False)
    required = {"date", "conference", "home_team", "away_team", "home_goals", "away_goals", "status", "source_url", "validation_note"}
    if not required.issubset(existing.columns):
        raise RuntimeError(f"results.csv is missing required columns: {sorted(required - set(existing.columns))}")
    imported = []
    errors = []
    for conference, url in SOURCES:
        try:
            rows = parse_page(fetch_page(url), conference, url)
            print(f"{conference}: parsed {len(rows)} scored results")
            imported.extend(rows)
        except Exception as exc:
            errors.append(f"{conference}: {exc}")
    if errors:
        raise RuntimeError("At least one source failed; preserving the current ledger. " + " | ".join(errors))
    if len(imported) < 5:
        raise RuntimeError(f"Only parsed {len(imported)} scored matches; threshold not met, preserving current ledger.")

    # Merge source scores by date/home/away, retaining user-confirmed notes where possible.
    # The scraper only replaces scores for the exact same fixture; it never deletes rows.
    out = existing.copy()
    key_cols = ["date", "home_team", "away_team"]
    for item in imported:
        mask = pd.Series(True, index=out.index)
        for col in key_cols:
            mask &= out[col].astype(str).eq(str(item[col]))
        if mask.any():
            idx = out.index[mask][0]
            old_status = str(out.at[idx, "status"])
            if "user confirmed" not in old_status.lower():
                for col in ["home_goals", "away_goals", "status", "source_url", "validation_note"]:
                    out.at[idx, col] = item[col]
                # Source-confirmed conference is kept only for same-conference fixtures.
                out.at[idx, "conference"] = item["conference"]
        else:
            out.loc[len(out)] = item
    out = out.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    # Keep user-confirmed cross-conference labels intact; normalize other rows from team mapping.
    team_map = pd.read_csv(TEAMS).set_index("team")["conference"].to_dict()
    for idx, row in out.iterrows():
        hconf, aconf = team_map.get(row.home_team), team_map.get(row.away_team)
        if hconf and aconf and hconf != aconf and "user confirmed" not in str(row.status).lower():
            out.at[idx, "conference"] = "Cross-conference"
        elif hconf and aconf and hconf == aconf and "user confirmed" not in str(row.status).lower():
            out.at[idx, "conference"] = f"GLA {hconf}"
    out.to_csv(RESULTS, index=False)
    print(f"Updated results.csv: {len(out)} total rows; {len(imported)} scraped source results.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
