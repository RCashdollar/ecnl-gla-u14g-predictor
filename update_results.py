"""Conservative refresh for ECNL-RL U14 Girls GLA.

The prior scraper inferred home/away teams by looking near score text. That is
unsafe for card-based pages: it can combine scores/teams from adjacent cards.
This version fails closed. It preserves the verified ledger, imports only
plausible same-conference results in the current season up to today's date,
never overwrites an existing fixture, and quarantines suspicious/conflicting
candidates for manual review. It is intentionally conservative rather than
pretending every scraped score is verified.
"""
from __future__ import annotations
import re, sys
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).parent
RESULTS=ROOT/'results.csv'
TEAMS=ROOT/'teams.csv'
QUARANTINE=ROOT/'scrape_quarantine.csv'
SOURCES=[('East','https://leaguetables.soccer/ecnl-rl-girls/u14/gla-east'),('West','https://leaguetables.soccer/ecnl-rl-girls/u14/gla-west')]
ALIASES={
 'Elite FC':'Elite FC ECNL RL G2012/13','Steel City FC':'Steel City FC ECNL RL G2012/13',
 'Cincinnati Elite FC':'Cincinnati Elite FC ECNL RL G2012/13','Columbus Force':'Columbus Force ECNL RL G2012/13',
 'FC Dayton':'FC Dayton ECNL RL G2012/13','T3FC':'T3FC ECNL RL G2012/13',
 'Cuyahoga Valley Soccer Academy':'Cuyahoga Valley Soccer Academy ECNL RL G2012/13','CVSA':'Cuyahoga Valley Soccer Academy ECNL RL G2012/13',
 'Northern Steel':'Northern Steel ECNL RL G2012/13','Manta':'Manta ECNL RL G2012/13','Canton Force':'Canton Force ECNL RL G2012/13',
 'WNY Flash Erie':'WNY Flash Erie ECNL RL G2012/13','WNY Flash Rochester':'WNY Flash Rochester ECNL RL G2012/13'}
ALIASES=dict(sorted(ALIASES.items(),key=lambda x:len(x[0]),reverse=True))
TEAM_RE=re.compile('|'.join(re.escape(x) for x in ALIASES),re.I)
SCORE_RE=re.compile(r'^(\d{1,2})\s*[-–—:]\s*(\d{1,2})$')
DATE_RE=re.compile(r'\b([A-Z][a-z]{2,8})\s+(\d{1,2}),\s+(20\d{2})\b')
COLS=['date','conference','home_team','away_team','home_goals','away_goals','status','source_url','validation_note']

def fetch(url):
 r=requests.get(url,timeout=25,headers={'User-Agent':'Mozilla/5.0 (compatible; GLA-U14G-results-refresh/2.0)'})
 r.raise_for_status()
 if len(r.text)<5000: raise RuntimeError(f'Unexpectedly short source page: {url}')
 return r.text

def canon(s):
 for a,c in ALIASES.items():
  if s.strip().casefold()==a.casefold(): return c
 return None

def parse(html, conf, url):
 soup=BeautifulSoup(html,'html.parser')
 tokens=[' '.join(s.split()) for s in soup.stripped_strings]
 found=[]; seen=set()
 # Parse only exact score tokens with a nearby pair of known teams and a nearby date.
 # Any result failing downstream checks is quarantined rather than merged.
 for i,t in enumerate(tokens):
  m=SCORE_RE.fullmatch(t.strip())
  if not m: continue
  before=[(j,canon(tokens[j])) for j in range(max(0,i-8),i) if canon(tokens[j])]
  after=[(j,canon(tokens[j])) for j in range(i+1,min(len(tokens),i+9)) if canon(tokens[j])]
  if not before or not after: continue
  # Closest known team on each side only; do not jump across a large page region.
  hi,home=max(before,key=lambda z:z[0]); ai,away=min(after,key=lambda z:z[0])
  if not home or not away or home==away: continue
  d=None
  for j in range(i,max(-1,i-35),-1):
   dm=DATE_RE.search(tokens[j])
   if dm:
    for fmt in ('%B %d %Y','%b %d %Y'):
     try: d=datetime.strptime(' '.join(dm.groups()),fmt).date().isoformat(); break
     except ValueError: pass
    if d: break
  if not d: continue
  key=(d,home,away,int(m.group(1)),int(m.group(2)))
  if key not in seen:
   seen.add(key); found.append(dict(date=d,conference=conf,home_team=home,away_team=away,home_goals=int(m.group(1)),away_goals=int(m.group(2)),status='Final — conservative auto-import',source_url=url,validation_note='Auto-import candidate; verify if disputed.'))
 return found

def main():
 if not RESULTS.exists() or not TEAMS.exists(): raise RuntimeError('results.csv or teams.csv is missing')
 old=pd.read_csv(RESULTS,keep_default_na=False)
 if not set(COLS).issubset(old.columns): raise RuntimeError('results.csv does not have the expected columns; no changes made')
 team_df=pd.read_csv(TEAMS,keep_default_na=False)
 confmap=dict(zip(team_df.team,team_df.conference))
 imported=[]
 for conf,url in SOURCES:
  rows=parse(fetch(url),conf,url); print(f'GLA {conf}: parsed {len(rows)} candidate score cards'); imported.extend(rows)
 today=date.today(); season_start=date(2026,7,1); accepted=[]; rejected=[]
 # Existing fixture identity is date + unordered teams, so reverse home/away duplicates are detected.
 def fixture_key(r): return (str(r['date']),frozenset([str(r['home_team']),str(r['away_team'])]))
 existing_by_key={}
 for _,r in old.iterrows(): existing_by_key.setdefault(fixture_key(r),[]).append(r)
 candidates={}
 for r in imported:
  reason=''
  try: d=date.fromisoformat(r['date'])
  except Exception: reason='Invalid date'
  if not reason and not (season_start<=d<=today): reason='Outside current season or future-dated'
  if not reason and (r['home_goals']>15 or r['away_goals']>15): reason='Implausible score; possible parsing error'
  hc,ac=confmap.get(r['home_team']),confmap.get(r['away_team'])
  if not reason and (not hc or not ac): reason='Team not in configured 12-team list'
  if not reason and (hc!=ac or hc!=r['conference']): reason='Conference/source mismatch; manual review required'
  key=fixture_key(r)
  if not reason and key in existing_by_key:
   # Existing ledger is authoritative; never overwrite it. Compare exact score if same orientation.
   prior=existing_by_key[key]
   same=any(str(x['home_team'])==r['home_team'] and str(x['away_team'])==r['away_team'] and str(x['home_goals'])==str(r['home_goals']) and str(x['away_goals'])==str(r['away_goals']) for x in prior)
   if not same: reason='Existing fixture found with different score/orientation; manual review required'
   else: continue
  if reason:
   rejected.append({**r,'quarantine_reason':reason})
  else:
   candidates.setdefault(key,[]).append(r)
 for key,rows in candidates.items():
  scores={(x['home_team'],x['away_team'],x['home_goals'],x['away_goals']) for x in rows}
  if len(scores)>1:
   rejected.extend({**r,'quarantine_reason':'Conflicting scraped candidates for same fixture'} for r in rows)
  else: accepted.append(rows[0])
 if rejected:
  q=pd.DataFrame(rejected)
  if QUARANTINE.exists():
   prior=pd.read_csv(QUARANTINE,keep_default_na=False)
   q=pd.concat([prior,q],ignore_index=True).drop_duplicates(subset=['date','home_team','away_team','home_goals','away_goals','quarantine_reason'])
  q.to_csv(QUARANTINE,index=False)
  print(f'Quarantined {len(rejected)} suspicious/conflicting candidates in scrape_quarantine.csv')
 if accepted:
  out=pd.concat([old,pd.DataFrame(accepted,columns=COLS)],ignore_index=True)
  out=out.drop_duplicates(subset=['date','home_team','away_team'],keep='first')
  out.to_csv(RESULTS,index=False)
 print(f'Added {len(accepted)} validated new results; preserved {len(old)} existing rows.')
 print('Existing results are never overwritten. Review scrape_quarantine.csv for rejected candidates.')

if __name__=='__main__':
 try: main()
 except Exception as e:
  print(f'ERROR: {e}',file=sys.stderr); sys.exit(1)
