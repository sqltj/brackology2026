# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1: Setup & Data Ingestion
# MAGIC Populate `bracketology.raw.*` Delta tables from ESPN APIs.

# COMMAND ----------

# MAGIC %pip install requests pandas
# MAGIC %restart_python

# COMMAND ----------

import requests
import pandas as pd
import time
import traceback as tb
from datetime import datetime, timedelta

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh)"}

def espn(path, params=None, retries=3):
    for i in range(retries):
        try:
            r = requests.get(f"{ESPN}/{path}", params=params, headers=HDR, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 ** i)
            else:
                return None

print("Ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Teams

# COMMAND ----------

teams = []
for page in range(1, 8):
    data = espn("teams", {"limit": 100, "page": page, "groups": 50})
    if not data:
        break
    before = len(teams)
    for sport in data.get("sports", []):
        for league in sport.get("leagues", []):
            for entry in league.get("teams", []):
                t = entry.get("team", entry)
                logos = t.get("logos") or []
                groups = t.get("groups") or {}
                if not isinstance(groups, dict):
                    groups = {}
                parent = groups.get("parent") or {}
                if not isinstance(parent, dict):
                    parent = {}
                teams.append({
                    "team_id": int(t.get("id", 0)),
                    "name": str(t.get("displayName", "")),
                    "abbreviation": str(t.get("abbreviation", "")),
                    "short_name": str(t.get("shortDisplayName", "")),
                    "color": str(t.get("color", "")),
                    "logo_url": str(logos[0]["href"]) if logos else "",
                    "conference_id": str(groups.get("id", "")),
                    "conference_name": str(parent.get("shortName", groups.get("shortName", ""))),
                })
    if len(teams) == before:
        break
    print(f"  Page {page}: {len(teams)} teams")

print(f"Total teams: {len(teams)}")
tdf = spark.createDataFrame(pd.DataFrame(teams))
tdf.write.mode("overwrite").saveAsTable("bracketology.raw.teams")
print(f"  Saved: {spark.table('bracketology.raw.teams').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Team Season Stats

# COMMAND ----------

team_ids = [r.team_id for r in spark.table("bracketology.raw.teams").select("team_id").collect()]
stats_list = []

for i, tid in enumerate(team_ids):
    if i % 100 == 0 and i > 0:
        print(f"  {i}/{len(team_ids)}...")
    try:
        data = espn(f"teams/{tid}/statistics")
        if not data:
            continue
        row = {"team_id": int(tid), "season": 2026}
        # Try multiple response shapes
        results = data.get("results", data.get("statistics", {}))
        if isinstance(results, list):
            results = results[0] if results else {}
        cats = []
        if isinstance(results, dict):
            cats = results.get("categories", [])
            if not cats:
                splits = results.get("splits", {})
                if isinstance(splits, dict):
                    cats = splits.get("categories", [])
        for cat in cats:
            cn = str(cat.get("name", "")).lower().replace(" ", "_").replace("-", "_")
            for s in cat.get("stats", []):
                sn = str(s.get("name", "")).lower().replace(" ", "_").replace("-", "_")
                key = f"{cn}_{sn}" if cn else sn
                key = ''.join(c if c.isalnum() or c == '_' else '_' for c in key)[:60]
                val = s.get("value", 0)
                try:
                    row[key] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    row[key] = 0.0
        stats_list.append(row)
    except Exception:
        pass
    time.sleep(0.05)

print(f"Stats for {len(stats_list)} teams")
if stats_list:
    pdf = pd.DataFrame(stats_list).fillna(0.0)
    for c in pdf.columns:
        if c not in ("team_id", "season"):
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0.0)
    spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable("bracketology.raw.team_season_stats")
    print(f"  Saved: {spark.table('bracketology.raw.team_season_stats').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Regular Season Games (2025-26)

# COMMAND ----------

def parse_scoreboard(data, season=2026, force_tourney=False):
    """Parse ESPN scoreboard JSON into list of dicts."""
    rows = []
    if not data or "events" not in data:
        return rows
    for ev in data.get("events", []):
        try:
            g = dict(game_id=str(ev.get("id","")), date=str(ev.get("date","")),
                     season=int(season), name=str(ev.get("name","")),
                     short_name=str(ev.get("shortName","")),
                     home_team_id=0, home_team_name="", home_score=0, home_winner=False,
                     away_team_id=0, away_team_name="", away_score=0, away_winner=False,
                     neutral_site=False, venue="", status="",
                     tournament_game=force_tourney, is_conference=False)
            for comp in ev.get("competitions", []):
                g["neutral_site"] = bool(comp.get("neutralSite", False))
                v = comp.get("venue")
                g["venue"] = str(v.get("fullName","")) if isinstance(v, dict) else ""
                for c in comp.get("competitors", []):
                    pfx = "home" if c.get("homeAway") == "home" else "away"
                    td = c.get("team", {})
                    g[f"{pfx}_team_id"] = int(td.get("id", 0))
                    g[f"{pfx}_team_name"] = str(td.get("displayName", ""))
                    try:
                        g[f"{pfx}_score"] = int(c.get("score", 0))
                    except (ValueError, TypeError):
                        g[f"{pfx}_score"] = 0
                    g[f"{pfx}_winner"] = bool(c.get("winner", False))
                for n in comp.get("notes", []):
                    h = str(n.get("headline", "")).lower()
                    if "conference" in h: g["is_conference"] = True
                    if "ncaa" in h or "tournament" in h: g["tournament_game"] = True
                st = comp.get("status", {})
                if isinstance(st, dict):
                    g["status"] = str(st.get("type", {}).get("description", ""))
            rows.append(g)
        except Exception:
            pass
    return rows

# COMMAND ----------

all_games = []
cur = datetime(2025, 11, 4)
end = datetime(2026, 3, 15)
while cur <= end:
    ds = cur.strftime("%Y%m%d")
    if cur.day == 1:
        print(f"  {cur.strftime('%b %Y')}: {len(all_games)} games so far")
    data = espn("scoreboard", {"dates": ds, "groups": 50, "limit": 200})
    all_games.extend(parse_scoreboard(data))
    cur += timedelta(days=1)
    time.sleep(0.03)

print(f"Regular season games: {len(all_games)}")
if all_games:
    spark.createDataFrame(pd.DataFrame(all_games)).write.mode("overwrite").saveAsTable("bracketology.raw.regular_season_games")
    print(f"  Saved: {spark.table('bracketology.raw.regular_season_games').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Historical Tournament Games (2016-2025)

# COMMAND ----------

RANGES = {
    2016:("20160315","20160405"), 2017:("20170314","20170404"),
    2018:("20180313","20180403"), 2019:("20190319","20190409"),
    2021:("20210318","20210406"), 2022:("20220315","20220405"),
    2023:("20230314","20230404"), 2024:("20240319","20240409"),
    2025:("20250318","20250408"),
}

all_tourney = []
for yr, (s, e) in RANGES.items():
    sd, ed = datetime.strptime(s,"%Y%m%d"), datetime.strptime(e,"%Y%m%d")
    yr_games = []
    c = sd
    while c <= ed:
        data = espn("scoreboard", {"dates": c.strftime("%Y%m%d"), "groups":50, "limit":200, "seasontype":3})
        yr_games.extend(parse_scoreboard(data, season=yr, force_tourney=True))
        c += timedelta(days=1)
        time.sleep(0.03)
    all_tourney.extend(yr_games)
    print(f"  {yr}: {len(yr_games)} games")

print(f"Total tourney games: {len(all_tourney)}")
if all_tourney:
    spark.createDataFrame(pd.DataFrame(all_tourney)).write.mode("overwrite").saveAsTable("bracketology.raw.historical_tourney")
    print(f"  Saved: {spark.table('bracketology.raw.historical_tourney').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Historical Seasons via Team Schedules

# COMMAND ----------

import random
random.seed(42)
sample_ids = random.sample(team_ids, min(80, len(team_ids)))
seasons_list = [2016,2017,2018,2019,2021,2022,2023,2024,2025]

all_hist = []
for season in seasons_list:
    seen = set()
    season_rows = []
    for tid in sample_ids:
        try:
            data = espn(f"teams/{tid}/schedule", {"season": season})
            if not data or "events" not in data:
                continue
            for ev in data["events"]:
                gid = str(ev.get("id",""))
                if not gid or gid in seen:
                    continue
                for comp in ev.get("competitions", []):
                    st_desc = ""
                    st_obj = comp.get("status", {})
                    if isinstance(st_obj, dict):
                        st_desc = str(st_obj.get("type",{}).get("description",""))
                    if st_desc != "Final":
                        continue
                    seen.add(gid)
                    ns = bool(comp.get("neutralSite", False))
                    stype = comp.get("seasonType", {})
                    st_int = int(stype.get("type",0)) if isinstance(stype, dict) else 0
                    comps = comp.get("competitors", [])
                    if len(comps) != 2:
                        continue
                    for ci, c in enumerate(comps):
                        td = c.get("team", {})
                        opp = comps[1-ci]
                        otd = opp.get("team", {})
                        try: sc = int(c.get("score",0))
                        except: sc = 0
                        try: osc = int(opp.get("score",0))
                        except: osc = 0
                        season_rows.append({
                            "game_id": gid, "date": str(ev.get("date","")),
                            "season": int(season), "team_id": int(td.get("id",0)),
                            "opponent_id": int(otd.get("id",0)),
                            "opponent_name": str(otd.get("displayName","")),
                            "score": sc, "opponent_score": osc,
                            "home_away": str(c.get("homeAway","")),
                            "winner": bool(c.get("winner", False)),
                            "neutral_site": ns, "status": st_desc, "season_type": st_int,
                        })
        except Exception:
            pass
        time.sleep(0.03)
    all_hist.extend(season_rows)
    print(f"  {season}: {len(seen)} games, {len(season_rows)} rows")

print(f"Total historical rows: {len(all_hist)}")
if all_hist:
    spark.createDataFrame(pd.DataFrame(all_hist)).write.mode("overwrite").saveAsTable("bracketology.raw.historical_seasons")
    print(f"  Saved: {spark.table('bracketology.raw.historical_seasons').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tournament Seeds

# COMMAND ----------

all_seeds = []
for yr, (s, e) in RANGES.items():
    sd, ed = datetime.strptime(s,"%Y%m%d"), datetime.strptime(e,"%Y%m%d")
    seen = set()
    c = sd
    while c <= ed:
        data = espn("scoreboard", {"dates": c.strftime("%Y%m%d"), "groups":50, "limit":200, "seasontype":3})
        if data:
            for ev in data.get("events", []):
                for comp in ev.get("competitions", []):
                    for cx in comp.get("competitors", []):
                        td = cx.get("team", {})
                        tid = int(td.get("id", 0))
                        if tid == 0 or tid in seen:
                            continue
                        sv = None
                        cr = cx.get("curatedRank")
                        if isinstance(cr, dict):
                            sv = cr.get("current")
                        if sv is None:
                            sv = cx.get("seed")
                        if sv is not None:
                            try:
                                si = int(sv)
                                if 1 <= si <= 16:
                                    seen.add(tid)
                                    all_seeds.append({"team_id":tid, "team_name":str(td.get("displayName","")), "seed":si, "season":yr})
                            except: pass
        c += timedelta(days=1)
        time.sleep(0.03)
    print(f"  {yr}: {len(seen)} seeded teams")

# 2026
seen26 = set()
for d in range(15, 28):
    data = espn("scoreboard", {"dates": f"202603{d:02d}", "groups":50, "limit":200, "seasontype":3})
    if data:
        for ev in data.get("events", []):
            for comp in ev.get("competitions", []):
                for cx in comp.get("competitors", []):
                    td = cx.get("team", {})
                    tid = int(td.get("id", 0))
                    if tid == 0 or tid in seen26: continue
                    sv = None
                    cr = cx.get("curatedRank")
                    if isinstance(cr, dict): sv = cr.get("current")
                    if sv is None: sv = cx.get("seed")
                    if sv is not None:
                        try:
                            si = int(sv)
                            if 1 <= si <= 16:
                                seen26.add(tid)
                                all_seeds.append({"team_id":tid, "team_name":str(td.get("displayName","")), "seed":si, "season":2026})
                        except: pass
    time.sleep(0.03)
print(f"  2026: {len(seen26)} seeded teams")

print(f"Total seeds: {len(all_seeds)}")
if all_seeds:
    spark.createDataFrame(pd.DataFrame(all_seeds)).write.mode("overwrite").saveAsTable("bracketology.raw.tourney_seeds")
    print(f"  Saved: {spark.table('bracketology.raw.tourney_seeds').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Current 2026 Tournament Results

# COMMAND ----------

ct = []
for m, days in [(3, range(17,32)), (4, range(1,10))]:
    for d in days:
        data = espn("scoreboard", {"dates": f"2026{m:02d}{d:02d}", "groups":50, "limit":200, "seasontype":3})
        ct.extend(parse_scoreboard(data, 2026, force_tourney=True))
        time.sleep(0.03)

print(f"2026 tourney games: {len(ct)}")
if ct:
    spark.createDataFrame(pd.DataFrame(ct)).write.mode("overwrite").saveAsTable("bracketology.raw.current_tourney_results")
else:
    # Empty table
    spark.createDataFrame(pd.DataFrame([{
        "game_id":"","date":"","season":2026,"name":"","short_name":"",
        "home_team_id":0,"home_team_name":"","home_score":0,"home_winner":False,
        "away_team_id":0,"away_team_name":"","away_score":0,"away_winner":False,
        "neutral_site":False,"venue":"","status":"","tournament_game":True,"is_conference":False
    }])).limit(0).write.mode("overwrite").saveAsTable("bracketology.raw.current_tourney_results")
print(f"  Saved: {spark.table('bracketology.raw.current_tourney_results').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

print("=" * 60)
print("RAW DATA INGESTION SUMMARY")
print("=" * 60)
for t in ["teams","team_season_stats","regular_season_games","historical_tourney","historical_seasons","tourney_seeds","current_tourney_results"]:
    try:
        full = f"bracketology.raw.{t}"
        c = spark.table(full).count()
        print(f"  {full}: {c:,} rows")
    except Exception as e:
        print(f"  bracketology.raw.{t}: ERROR - {e}")
print("=" * 60)
