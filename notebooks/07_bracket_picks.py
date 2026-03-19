# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 7: Full Bracket Picks — Calibrated Predictions
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Maps the **actual 2026 NCAA Tournament bracket** (all 4 regions)
# MAGIC 2. Applies **calibration correction** to fix mid-major bias
# MAGIC 3. Generates picks for **every game** through the Championship
# MAGIC 4. Runs Monte Carlo simulation on the real bracket structure
# MAGIC 5. Tracks actual results as games complete
# MAGIC
# MAGIC ### Why Calibration Is Needed
# MAGIC
# MAGIC Our ensemble (90% XGBoost) over-indexes on win percentage without adequately
# MAGIC penalizing weak strength of schedule. This causes mid-majors like High Point (30-3
# MAGIC in the Big South) to look better than Wisconsin (22-10 in the Big Ten). We fix this
# MAGIC by blending model predictions with historical seed-based priors.

# COMMAND ----------

# MAGIC %pip install xgboost scikit-learn pandas numpy requests
# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import json
import requests
import time
from datetime import datetime

np.random.seed(42)

def safe_int(v, default=0):
    if isinstance(v, dict): return default
    try: return int(v)
    except: return default

def safe_float(v, default=0.0):
    if isinstance(v, dict): return default
    try: return float(v)
    except: return default

print("Imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define the Actual 2026 Bracket
# MAGIC
# MAGIC Sourced from ESPN bracket page. First Four results already incorporated.

# COMMAND ----------

# First Four results (completed Mar 17-18):
#   Texas 68, NC State 66 → Texas is 11-seed in WEST
#   Miami OH 89, SMU 79 → Miami OH is 11-seed in MIDWEST
#   Howard 86, UMBC 83 → Howard is 16-seed in MIDWEST
#   Prairie View 67, Lehigh 55 → Prairie View is 16-seed in SOUTH

# Actual bracket matchups by region
# Each tuple: (seed_a, team_a_name, seed_b, team_b_name)
BRACKET = {
    "EAST": [
        (1, "Duke", 16, "Siena"),
        (8, "Ohio State", 9, "TCU"),
        (5, "St. John's", 12, "Northern Iowa"),
        (4, "Kansas", 13, "CA Baptist"),
        (6, "Louisville", 11, "South Florida"),
        (3, "Michigan State", 14, "North Dakota State"),
        (7, "UCLA", 10, "UCF"),
        (2, "UConn", 15, "Furman"),
    ],
    "WEST": [
        (1, "Arizona", 16, "Long Island"),
        (8, "Villanova", 9, "Utah State"),
        (5, "Wisconsin", 12, "High Point"),
        (4, "Arkansas", 13, "Hawai'i"),
        (6, "BYU", 11, "Texas"),
        (3, "Gonzaga", 14, "Kennesaw State"),
        (7, "Miami", 10, "Missouri"),
        (2, "Purdue", 15, "Queens"),
    ],
    "SOUTH": [
        (1, "Florida", 16, "Prairie View"),
        (8, "Clemson", 9, "Iowa"),
        (5, "Vanderbilt", 12, "McNeese"),
        (4, "Nebraska", 13, "Troy"),
        (6, "North Carolina", 11, "VCU"),
        (3, "Illinois", 14, "Penn"),
        (7, "Saint Mary's", 10, "Texas A&M"),
        (2, "Houston", 15, "Idaho"),
    ],
    "MIDWEST": [
        (1, "Michigan", 16, "Howard"),
        (8, "Georgia", 9, "Saint Louis"),
        (5, "Texas Tech", 12, "Akron"),
        (4, "Alabama", 13, "Hofstra"),
        (6, "Tennessee", 11, "Miami OH"),
        (3, "Virginia", 14, "Wright State"),
        (7, "Kentucky", 10, "Santa Clara"),
        (2, "Iowa State", 15, "Tennessee State"),
    ],
}

# Map display names to DB team names for lookup
NAME_MAP = {
    "Duke": "Duke Blue Devils",
    "Siena": "Siena Saints",
    "Ohio State": "Ohio State Buckeyes",
    "TCU": "TCU Horned Frogs",
    "St. John's": "St. John's Red Storm",
    "Northern Iowa": "Northern Iowa Panthers",
    "Kansas": "Kansas Jayhawks",
    "CA Baptist": "California Baptist Lancers",
    "Louisville": "Louisville Cardinals",
    "South Florida": "South Florida Bulls",
    "Michigan State": "Michigan State Spartans",
    "North Dakota State": "North Dakota State Bison",
    "UCLA": "UCLA Bruins",
    "UCF": "UCF Knights",
    "UConn": "UConn Huskies",
    "Furman": "Furman Paladins",
    "Arizona": "Arizona Wildcats",
    "Long Island": "Long Island University Sharks",
    "Villanova": "Villanova Wildcats",
    "Utah State": "Utah State Aggies",
    "Wisconsin": "Wisconsin Badgers",
    "High Point": "High Point Panthers",
    "Arkansas": "Arkansas Razorbacks",
    "Hawai'i": "Hawai'i Rainbow Warriors",
    "BYU": "BYU Cougars",
    "Texas": "Texas Longhorns",
    "Gonzaga": "Gonzaga Bulldogs",
    "Kennesaw State": "Kennesaw State Owls",
    "Miami": "Miami Hurricanes",
    "Missouri": "Missouri Tigers",
    "Purdue": "Purdue Boilermakers",
    "Queens": "Queens University Royals",
    "Florida": "Florida Gators",
    "Prairie View": "Prairie View A&M Panthers",
    "Clemson": "Clemson Tigers",
    "Iowa": "Iowa Hawkeyes",
    "Vanderbilt": "Vanderbilt Commodores",
    "McNeese": "McNeese Cowboys",
    "Nebraska": "Nebraska Cornhuskers",
    "Troy": "Troy Trojans",
    "North Carolina": "North Carolina Tar Heels",
    "VCU": "VCU Rams",
    "Illinois": "Illinois Fighting Illini",
    "Penn": "Pennsylvania Quakers",
    "Saint Mary's": "Saint Mary's Gaels",
    "Texas A&M": "Texas A&M Aggies",
    "Houston": "Houston Cougars",
    "Idaho": "Idaho Vandals",
    "Michigan": "Michigan Wolverines",
    "Howard": "Howard Bison",
    "Georgia": "Georgia Bulldogs",
    "Saint Louis": "Saint Louis Billikens",
    "Texas Tech": "Texas Tech Red Raiders",
    "Akron": "Akron Zips",
    "Alabama": "Alabama Crimson Tide",
    "Hofstra": "Hofstra Pride",
    "Tennessee": "Tennessee Volunteers",
    "Miami OH": "Miami (OH) RedHawks",
    "Virginia": "Virginia Cavaliers",
    "Wright State": "Wright State Raiders",
    "Kentucky": "Kentucky Wildcats",
    "Santa Clara": "Santa Clara Broncos",
    "Iowa State": "Iowa State Cyclones",
    "Tennessee State": "Tennessee State Tigers",
}

total_games = sum(len(v) for v in BRACKET.values())
print(f"Bracket loaded: {len(BRACKET)} regions, {total_games} first-round games")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pull Defense & Rebounding Stats from ESPN
# MAGIC
# MAGIC Our feature table lacks rebounding and detailed defensive stats.
# MAGIC Pull them live from ESPN's team statistics endpoint for every tournament team.

# COMMAND ----------

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh)"}

# Get team IDs from our DB
teams_db = spark.table("bracketology.raw.teams").toPandas()
team_id_lookup = {}
for _, row in teams_db.iterrows():
    team_id_lookup[str(row["name"])] = safe_int(row["team_id"])

# Also get team_season_features for existing stats
tsf = spark.table("bracketology.features.team_season_features").filter("season = 2026").toPandas()
tsf_lookup = {}
for _, row in tsf.iterrows():
    tsf_lookup[safe_int(row["team_id"])] = row.to_dict()

# Pull stats from ESPN for each tournament team
defense_stats = {}  # team_name -> {ppg_allowed, reb_per_game, opp_fg_pct, blocks, steals, def_reb}

all_bracket_teams = set()
for matchups in BRACKET.values():
    for sa, ta, sb, tb in matchups:
        all_bracket_teams.add(ta)
        all_bracket_teams.add(tb)

print(f"Pulling ESPN stats for {len(all_bracket_teams)} tournament teams...")
pulled = 0
failed = 0

for team_name in sorted(all_bracket_teams):
    db_name = NAME_MAP.get(team_name, team_name)
    tid = team_id_lookup.get(db_name, 0)
    if tid == 0:
        # Try partial match
        for k, v in team_id_lookup.items():
            if team_name.lower() in k.lower():
                tid = v
                break

    if tid == 0:
        failed += 1
        defense_stats[team_name] = {"ppg_allowed": 70.0, "reb_margin": 0.0, "def_reb": 25.0,
                                     "opp_fg_pct": 0.44, "steals": 6.0, "blocks": 3.0}
        continue

    try:
        url = f"{ESPN_BASE}/teams/{tid}/statistics"
        resp = requests.get(url, headers=HDR, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        stats = {}
        # Parse stats from ESPN response
        for cat in data.get("results", data.get("statistics", {}).get("splits", {}).get("categories", [])):
            if isinstance(cat, dict):
                cat_name = cat.get("name", cat.get("displayName", ""))
                for stat in cat.get("stats", []):
                    stat_name = stat.get("name", stat.get("abbreviation", ""))
                    stat_val = stat.get("value", stat.get("displayValue", "0"))
                    stats[stat_name] = safe_float(stat_val)

        # Also try alternate parsing (ESPN has multiple response formats)
        if not stats:
            for split in data.get("splits", {}).get("categories", []):
                for stat in split.get("stats", []):
                    stats[stat.get("name", "")] = safe_float(stat.get("value", 0))

        # Extract what we need
        ppg_allowed = safe_float(stats.get("avgPointsAgainst", stats.get("pointsAgainst", 0)))
        if ppg_allowed == 0:
            # Get from our features table
            feat = tsf_lookup.get(tid, {})
            ppg_allowed = safe_float(feat.get("avg_pts_against", 70.0))

        # Rebounding stats
        total_reb = safe_float(stats.get("avgRebounds", stats.get("totalRebounds", stats.get("rebounds", 0))))
        def_reb = safe_float(stats.get("avgDefensiveRebounds", stats.get("defensiveRebounds", 0)))
        reb_margin = safe_float(stats.get("reboundMargin", stats.get("avgReboundMargin", 0)))
        blocks = safe_float(stats.get("avgBlocks", stats.get("blocks", 0)))
        steals = safe_float(stats.get("avgSteals", stats.get("steals", 0)))
        opp_fg_pct = safe_float(stats.get("opponentFieldGoalPct", stats.get("fieldGoalPctDefense", 0.44)))

        defense_stats[team_name] = {
            "ppg_allowed": ppg_allowed if ppg_allowed > 0 else 70.0,
            "reb_margin": reb_margin,
            "def_reb": def_reb if def_reb > 0 else 25.0,
            "opp_fg_pct": opp_fg_pct if 0.2 < opp_fg_pct < 0.6 else 0.44,
            "steals": steals,
            "blocks": blocks,
        }
        pulled += 1
        time.sleep(0.1)

    except Exception as e:
        # Fall back to features table
        feat = tsf_lookup.get(tid, {})
        defense_stats[team_name] = {
            "ppg_allowed": safe_float(feat.get("avg_pts_against", 70.0)),
            "reb_margin": 0.0, "def_reb": 25.0,
            "opp_fg_pct": 0.44, "steals": 6.0, "blocks": 3.0,
        }
        failed += 1

print(f"Stats pulled: {pulled} success, {failed} fallback")

# Show defensive rankings of tournament teams
def_df = pd.DataFrame([
    {"team": t, **s} for t, s in defense_stats.items()
]).sort_values("ppg_allowed")

print("\n" + "=" * 75)
print("DEFENSIVE RANKINGS — Tournament Teams (Points Allowed Per Game)")
print("=" * 75)
for i, (_, row) in enumerate(def_df.head(15).iterrows()):
    print(f"  {i+1:>2}. {row['team']:<25s} PPG allowed: {row['ppg_allowed']:>5.1f}  "
          f"Reb margin: {row['reb_margin']:>+5.1f}  Blocks: {row['blocks']:>4.1f}")

print("\n  ...")
print("  WORST DEFENSES:")
for _, row in def_df.tail(5).iterrows():
    print(f"      {row['team']:<25s} PPG allowed: {row['ppg_allowed']:>5.1f}  "
          f"Reb margin: {row['reb_margin']:>+5.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Raw Model Predictions

# COMMAND ----------

# Load pairwise probabilities
pairwise_df = spark.table("bracketology.predictions.pairwise_probabilities").toPandas()

# Build lookup by team name
prob_lookup = {}
for _, row in pairwise_df.iterrows():
    a_name = str(row["team_a_name"])
    b_name = str(row["team_b_name"])
    prob_lookup[(a_name, b_name)] = float(row["p_team_a_wins"])
    prob_lookup[(b_name, a_name)] = float(row["p_team_b_wins"])

def get_raw_prob(team_a, team_b):
    """Get raw model P(team_a wins) from pairwise lookup."""
    db_a = NAME_MAP.get(team_a, team_a)
    db_b = NAME_MAP.get(team_b, team_b)
    p = prob_lookup.get((db_a, db_b))
    if p is not None:
        return p
    # Try reverse
    p = prob_lookup.get((db_b, db_a))
    if p is not None:
        return 1.0 - p
    return 0.5  # fallback

# Test
test_p = get_raw_prob("Duke", "Siena")
print(f"Test: Duke vs Siena raw prob = {test_p:.3f}")
print(f"Lookup has {len(prob_lookup)} entries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Calibration — Fixing Mid-Major Bias
# MAGIC
# MAGIC ### The Problem
# MAGIC Our XGBoost model (90% ensemble weight) treats `win_pct_diff` as a top feature.
# MAGIC A mid-major going 30-3 against weak opponents gets the same credit as a power
# MAGIC conference team going 28-5 against ranked opponents.
# MAGIC
# MAGIC ### The Fix
# MAGIC Blend model predictions with **historical seed-based win rates** as a Bayesian prior.
# MAGIC This anchors predictions to what actually happens in March — a 12-seed beats a 5-seed
# MAGIC ~35% of the time, not 95%.
# MAGIC
# MAGIC `P_calibrated = α × P_model + (1-α) × P_seed_history`
# MAGIC
# MAGIC We use α=0.4 (40% model, 60% seed prior) because our model has known calibration issues.
# MAGIC A well-calibrated model would use α=0.7+.

# COMMAND ----------

# Historical first-round win rates for the HIGHER seed (1985-2025)
# Source: NCAA tournament historical data
SEED_WIN_RATES = {
    (1, 16): 0.988,   # 1-seeds win 98.8%
    (2, 15): 0.937,   # 2-seeds win 93.7%
    (3, 14): 0.849,   # 3-seeds win 84.9%
    (4, 13): 0.786,   # 4-seeds win 78.6%
    (5, 12): 0.643,   # 5-seeds win 64.3%
    (6, 11): 0.628,   # 6-seeds win 62.8%
    (7, 10): 0.605,   # 7-seeds win 60.5%
    (8, 9):  0.512,   # 8-seeds win 51.2%
}

# For later rounds, use a simpler seed-differential model
def seed_prior(seed_a, seed_b):
    """Historical P(lower-numbered seed wins) for any seed matchup."""
    if seed_a == seed_b:
        return 0.5
    higher, lower = min(seed_a, seed_b), max(seed_a, seed_b)
    key = (higher, lower)
    if key in SEED_WIN_RATES:
        base = SEED_WIN_RATES[key]
    else:
        # General formula: each seed difference = ~3.3% advantage
        diff = lower - higher
        base = 0.5 + diff * 0.033
        base = min(base, 0.95)
    # Return from perspective of team_a
    return base if seed_a < seed_b else (1.0 - base)

# Calibration blend
ALPHA = 0.4  # Model weight (lower = more conservative, higher = trust model more)

def defense_rebound_adjustment(team_a, team_b):
    """
    Compute an adjustment factor based on defensive stats and rebounding.
    Returns a value centered at 0: positive favors team_a, negative favors team_b.

    Key insight: in March, defense and rebounding win. Teams that:
    - Allow fewer points per game (defensive efficiency)
    - Outrebound opponents (second-chance points, possession control)
    - Block more shots (rim protection)
    tend to outperform their seed in the tournament.
    """
    da = defense_stats.get(team_a, {})
    db = defense_stats.get(team_b, {})

    # Points allowed differential (lower is better, so flip sign)
    ppg_diff = safe_float(db.get("ppg_allowed", 70)) - safe_float(da.get("ppg_allowed", 70))
    # Normalize: 10 PPG difference ≈ max adjustment
    ppg_adj = np.clip(ppg_diff / 20.0, -0.15, 0.15)

    # Rebound margin differential
    reb_diff = safe_float(da.get("reb_margin", 0)) - safe_float(db.get("reb_margin", 0))
    reb_adj = np.clip(reb_diff / 15.0, -0.10, 0.10)

    # Blocks differential (rim protection)
    blk_diff = safe_float(da.get("blocks", 3)) - safe_float(db.get("blocks", 3))
    blk_adj = np.clip(blk_diff / 8.0, -0.05, 0.05)

    # Total adjustment (defense-weighted)
    return ppg_adj + reb_adj + blk_adj

def calibrated_prob(team_a, seed_a, team_b, seed_b):
    """Blend model prediction with seed-based prior + defense/rebounding adjustment."""
    p_model = get_raw_prob(team_a, team_b)
    p_prior = seed_prior(seed_a, seed_b)

    # Base calibration: blend model with seed prior
    p_cal = ALPHA * p_model + (1 - ALPHA) * p_prior

    # Defense & rebounding adjustment
    def_adj = defense_rebound_adjustment(team_a, team_b)
    p_cal += def_adj

    return float(np.clip(p_cal, 0.01, 0.99))

# Show calibration effect on a few key matchups
print("=" * 85)
print("CALIBRATION COMPARISON — Raw Model vs Calibrated (with Defense + Rebounding)")
print("=" * 85)
print(f"{'Matchup':<40s} {'Raw':>7} {'Prior':>7} {'Def Adj':>8} {'Final':>7}")
print("-" * 85)
examples = [
    ("Duke", 1, "Siena", 16),
    ("Wisconsin", 5, "High Point", 12),
    ("Michigan State", 3, "North Dakota State", 14),
    ("Tennessee", 6, "Miami OH", 11),
    ("Kansas", 4, "CA Baptist", 13),
    ("Kentucky", 7, "Santa Clara", 10),
    ("Ohio State", 8, "TCU", 9),
    ("North Carolina", 6, "VCU", 11),
    ("Alabama", 4, "Hofstra", 13),
]
for ta, sa, tb, sb in examples:
    raw = get_raw_prob(ta, tb)
    prior = seed_prior(sa, sb)
    def_adj = defense_rebound_adjustment(ta, tb)
    cal = calibrated_prob(ta, sa, tb, sb)
    label = f"[{sa}] {ta} vs [{sb}] {tb}"
    print(f"  {label:<38s} {raw:>6.1%} {prior:>6.1%} {def_adj:>+7.1%} {cal:>6.1%}")
print("-" * 85)
print(f"  α = {ALPHA} (model weight) + defense/rebounding adjustment")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. First Round Picks — All 32 Games

# COMMAND ----------

def pick_winner(seed_a, team_a, seed_b, team_b):
    """Return (winner_name, winner_seed, probability)."""
    p = calibrated_prob(team_a, seed_a, team_b, seed_b)
    if p >= 0.5:
        return team_a, seed_a, p
    else:
        return team_b, seed_b, 1.0 - p

first_round_picks = {}  # region -> list of (winner, winner_seed, prob)

for region in ["EAST", "WEST", "SOUTH", "MIDWEST"]:
    print("=" * 75)
    print(f"  {region} REGION — FIRST ROUND")
    print("=" * 75)
    print(f"  {'Matchup':<42s} {'Pick':<22s} {'Prob':>6} {'Upset?':>6}")
    print("-" * 75)

    region_picks = []
    for seed_a, team_a, seed_b, team_b in BRACKET[region]:
        winner, w_seed, prob = pick_winner(seed_a, team_a, seed_b, team_b)
        is_upset = "⚠️" if w_seed > min(seed_a, seed_b) else ""
        label = f"[{seed_a}] {team_a} vs [{seed_b}] {team_b}"
        print(f"  {label:<42s} {winner:<22s} {prob:>5.1%} {is_upset:>5}")
        region_picks.append((winner, w_seed, prob))

    first_round_picks[region] = region_picks
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Full Bracket — Through the Championship

# COMMAND ----------

def simulate_round(matchups, round_name):
    """Given list of (seed, team) tuples, pick winners for each pair."""
    results = []
    for i in range(0, len(matchups) - 1, 2):
        seed_a, team_a = matchups[i]
        seed_b, team_b = matchups[i + 1]
        winner, w_seed, prob = pick_winner(seed_a, team_a, seed_b, team_b)
        is_upset = "⚠️" if w_seed > min(seed_a, seed_b) else ""
        label = f"[{seed_a}] {team_a} vs [{seed_b}] {team_b}"
        print(f"  {label:<42s} → [{w_seed}] {winner:<20s} {prob:>5.1%} {is_upset}")
        results.append((w_seed, winner))
    return results

ROUND_NAMES = ["ROUND OF 32", "SWEET 16", "ELITE 8"]

region_winners = {}
for region in ["EAST", "WEST", "SOUTH", "MIDWEST"]:
    print("=" * 75)
    print(f"  {region} REGION")
    print("=" * 75)

    # Start with first round winners
    current = [(ws, wn) for wn, ws, _ in first_round_picks[region]]

    for rd_name in ROUND_NAMES:
        print(f"\n  --- {rd_name} ---")
        current = simulate_round(current, rd_name)

    region_winners[region] = current[0] if current else (0, "TBD")
    print(f"\n  🏆 {region} CHAMPION: [{current[0][0]}] {current[0][1]}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Final Four & Championship

# COMMAND ----------

print("=" * 75)
print("  FINAL FOUR")
print("=" * 75)

# Traditional Final Four pairings: EAST vs WEST, SOUTH vs MIDWEST
ff_game1 = (region_winners["EAST"], region_winners["WEST"])
ff_game2 = (region_winners["SOUTH"], region_winners["MIDWEST"])

print(f"\n  --- SEMIFINAL 1: EAST vs WEST ---")
seed_a, team_a = ff_game1[0]
seed_b, team_b = ff_game1[1]
w1, ws1, p1 = pick_winner(seed_a, team_a, seed_b, team_b)
print(f"  [{seed_a}] {team_a} vs [{seed_b}] {team_b}  →  [{ws1}] {w1} ({p1:.1%})")

print(f"\n  --- SEMIFINAL 2: SOUTH vs MIDWEST ---")
seed_a, team_a = ff_game2[0]
seed_b, team_b = ff_game2[1]
w2, ws2, p2 = pick_winner(seed_a, team_a, seed_b, team_b)
print(f"  [{seed_a}] {team_a} vs [{seed_b}] {team_b}  →  [{ws2}] {w2} ({p2:.1%})")

print(f"\n  --- CHAMPIONSHIP ---")
champ, cs, cp = pick_winner(ws1, w1, ws2, w2)
print(f"  [{ws1}] {w1} vs [{ws2}] {w2}  →  [{cs}] {champ} ({cp:.1%})")

print(f"\n{'=' * 75}")
print(f"  🏆 PREDICTED 2026 NCAA CHAMPION: [{cs}] {champ.upper()}")
print(f"{'=' * 75}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Monte Carlo Bracket Simulation (Calibrated)
# MAGIC
# MAGIC Rather than picking deterministic winners, simulate 10,000 brackets
# MAGIC using calibrated probabilities to get championship odds.

# COMMAND ----------

N_SIMS = 10000

# Track advancement counts
all_teams = []
for region, matchups in BRACKET.items():
    for sa, ta, sb, tb in matchups:
        all_teams.append((ta, sa, region))
        all_teams.append((tb, sb, region))

team_info = {t[0]: {"seed": t[1], "region": t[2]} for t in all_teams}
rounds = ["R32", "S16", "E8", "F4", "Final", "Champion"]
advance_counts = {t: {r: 0 for r in rounds} for t in team_info}

for sim in range(N_SIMS):
    # Simulate each region
    ff_teams = []
    for region in ["EAST", "WEST", "SOUTH", "MIDWEST"]:
        # First round
        current = []
        for sa, ta, sb, tb in BRACKET[region]:
            p = calibrated_prob(ta, sa, tb, sb)
            winner = ta if np.random.random() < p else tb
            w_seed = sa if winner == ta else sb
            current.append((w_seed, winner))
            advance_counts[winner]["R32"] += 1

        # R32
        next_round = []
        for i in range(0, len(current) - 1, 2):
            sa, ta = current[i]
            sb, tb = current[i + 1]
            p = calibrated_prob(ta, sa, tb, sb)
            winner = ta if np.random.random() < p else tb
            w_seed = sa if winner == ta else sb
            next_round.append((w_seed, winner))
            advance_counts[winner]["S16"] += 1
        current = next_round

        # S16
        next_round = []
        for i in range(0, len(current) - 1, 2):
            sa, ta = current[i]
            sb, tb = current[i + 1]
            p = calibrated_prob(ta, sa, tb, sb)
            winner = ta if np.random.random() < p else tb
            w_seed = sa if winner == ta else sb
            next_round.append((w_seed, winner))
            advance_counts[winner]["E8"] += 1
        current = next_round

        # E8 (region final)
        if len(current) >= 2:
            sa, ta = current[0]
            sb, tb = current[1]
            p = calibrated_prob(ta, sa, tb, sb)
            winner = ta if np.random.random() < p else tb
            w_seed = sa if winner == ta else sb
            advance_counts[winner]["F4"] += 1
            ff_teams.append((w_seed, winner))

    # Final Four
    if len(ff_teams) >= 4:
        # EAST vs WEST
        sa, ta = ff_teams[0]
        sb, tb = ff_teams[1]
        p = calibrated_prob(ta, sa, tb, sb)
        f1 = ta if np.random.random() < p else tb
        f1s = sa if f1 == ta else sb
        advance_counts[f1]["Final"] += 1

        # SOUTH vs MIDWEST
        sa, ta = ff_teams[2]
        sb, tb = ff_teams[3]
        p = calibrated_prob(ta, sa, tb, sb)
        f2 = ta if np.random.random() < p else tb
        f2s = sa if f2 == ta else sb
        advance_counts[f2]["Final"] += 1

        # Championship
        p = calibrated_prob(f1, f1s, f2, f2s)
        champ = f1 if np.random.random() < p else f2
        advance_counts[champ]["Champion"] += 1

# Build results table
sim_rows = []
for team, info in team_info.items():
    row = {"team_name": team, "seed": info["seed"], "region": info["region"]}
    for rd in rounds:
        row[f"p_{rd.lower()}"] = advance_counts[team][rd] / N_SIMS
    sim_rows.append(row)

sim_df = pd.DataFrame(sim_rows).sort_values("p_champion", ascending=False)

print("=" * 90)
print("CALIBRATED CHAMPIONSHIP ODDS (10,000 Simulations)")
print("=" * 90)
print(f"{'Seed':>4}  {'Team':<25s} {'Region':<8s} {'R32':>6} {'S16':>6} {'E8':>6} {'F4':>6} {'Final':>6} {'Champ':>7}")
print("-" * 90)
for _, t in sim_df.head(25).iterrows():
    print(f"[{int(t['seed']):2d}]  {t['team_name']:<25s} {t['region']:<8s} "
          f"{t['p_r32']:.1%}  {t['p_s16']:.1%}  {t['p_e8']:.1%}  "
          f"{t['p_f4']:.1%}  {t['p_final']:.1%}  {t['p_champion']:.1%}")
print(f"\nTotal champion prob: {sim_df['p_champion'].sum():.4f}")

# COMMAND ----------

# Save calibrated simulation to DB
spark.createDataFrame(sim_df).write.mode("overwrite").saveAsTable(
    "bracketology.predictions.bracket_simulation_calibrated"
)
print("Saved calibrated simulation to bracketology.predictions.bracket_simulation_calibrated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Upset Watch

# COMMAND ----------

print("=" * 75)
print("UPSET WATCH — First Round Games Where Underdog Has Real Chance")
print("=" * 75)

upsets = []
for region, matchups in BRACKET.items():
    for sa, ta, sb, tb in matchups:
        if sa == sb:
            continue
        higher_seed = ta if sa < sb else tb
        lower_seed = tb if sa < sb else ta
        s_high = min(sa, sb)
        s_low = max(sa, sb)
        p_upset = calibrated_prob(lower_seed, s_low, higher_seed, s_high)
        if p_upset > 0.20 and s_low - s_high >= 2:
            upsets.append({
                "region": region,
                "fav": f"[{s_high}] {higher_seed}",
                "dog": f"[{s_low}] {lower_seed}",
                "p_upset": p_upset,
                "diff": s_low - s_high,
            })

upsets.sort(key=lambda x: x["p_upset"], reverse=True)
for u in upsets:
    bar = "█" * int(u["p_upset"] * 40)
    print(f"  {u['region']:<8s} {u['dog']:<30s} over {u['fav']:<25s} {u['p_upset']:.1%} {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model vs Vegas Lines

# COMMAND ----------

# Vegas spreads from ESPN bracket page (as of Mar 18)
VEGAS_LINES = {
    ("Duke", "Siena"): -27.5,
    ("Ohio State", "TCU"): -2.5,
    ("St. John's", "Northern Iowa"): -9.5,
    ("Kansas", "CA Baptist"): -14.5,
    ("Louisville", "South Florida"): -4.5,
    ("Michigan State", "North Dakota State"): -16.5,
    ("UCLA", "UCF"): -5.5,
    ("UConn", "Furman"): -20.5,
    ("Arizona", "Long Island"): -30.5,
    ("Villanova", "Utah State"): 1.5,   # Utah State favored
    ("Wisconsin", "High Point"): -10.5,
    ("Arkansas", "Hawai'i"): -15.5,
    ("BYU", "Texas"): -2.5,
    ("Gonzaga", "Kennesaw State"): -21.5,
    ("Miami", "Missouri"): -1.5,
    ("Purdue", "Queens"): -25.5,
    ("Florida", "Prairie View"): -35.5,
    ("Clemson", "Iowa"): 2.5,   # Iowa favored
    ("Vanderbilt", "McNeese"): -11.5,
    ("Nebraska", "Troy"): -12.5,
    ("North Carolina", "VCU"): -2.5,
    ("Illinois", "Penn"): -25.5,
    ("Saint Mary's", "Texas A&M"): -3.5,
    ("Houston", "Idaho"): -23.5,
    ("Michigan", "Howard"): -31.5,
    ("Georgia", "Saint Louis"): -2.5,
    ("Texas Tech", "Akron"): -7.5,
    ("Alabama", "Hofstra"): -11.5,
    ("Tennessee", "Miami OH"): -10.5,
    ("Virginia", "Wright State"): -18.5,
    ("Kentucky", "Santa Clara"): -3.5,
    ("Iowa State", "Tennessee State"): -24.5,
}

def spread_to_prob(spread):
    """Convert point spread to implied win probability (empirical formula)."""
    return 1.0 / (1.0 + 10 ** (spread / 12.0))

print("=" * 90)
print("MODEL vs VEGAS — First Round")
print("=" * 90)
print(f"  {'Matchup':<40s} {'Model':>8} {'Vegas':>8} {'Agree?':>7}")
print("-" * 90)

agree_count = 0
total = 0
for region, matchups in BRACKET.items():
    for sa, ta, sb, tb in matchups:
        p_model = calibrated_prob(ta, sa, tb, sb)
        model_pick = ta if p_model >= 0.5 else tb

        spread = VEGAS_LINES.get((ta, tb), 0)
        # Negative spread means team_a is favored
        p_vegas = 1.0 - spread_to_prob(spread) if spread != 0 else 0.5
        vegas_pick = ta if p_vegas >= 0.5 else tb

        agree = "✅" if model_pick == vegas_pick else "❌"
        if model_pick == vegas_pick:
            agree_count += 1
        total += 1

        label = f"[{sa}] {ta} vs [{sb}] {tb}"
        print(f"  {label:<40s} {p_model:>7.1%} {p_vegas:>7.1%} {agree:>6}")

print(f"\nModel agrees with Vegas on {agree_count}/{total} games ({agree_count/total:.0%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Track Actual Results
# MAGIC
# MAGIC Run this section after games complete to see how predictions performed.

# COMMAND ----------

# Pull actual results from the current_tourney_results table
try:
    results_df = spark.table("bracketology.raw.current_tourney_results").toPandas()
    completed = results_df[results_df["status"] == "Final"]

    if len(completed) > 0:
        print("=" * 85)
        print("PREDICTION SCORECARD — Completed Games")
        print("=" * 85)
        print(f"  {'Game':<45s} {'Pick':>15} {'Result':>10} {'Correct':>8}")
        print("-" * 85)

        correct = 0
        total_scored = 0
        log_losses = []

        for _, game in completed.iterrows():
            home = str(game.get("home_team_name", ""))
            away = str(game.get("away_team_name", ""))
            home_score = safe_int(game.get("home_score", 0))
            away_score = safe_int(game.get("away_score", 0))

            # Find this game in our bracket
            p = get_raw_prob(home, away)  # Use raw for scoring (what we actually predicted)
            model_pick = home if p >= 0.5 else away
            actual_winner = home if home_score > away_score else away

            is_correct = model_pick == actual_winner
            if is_correct:
                correct += 1
            total_scored += 1

            # Log loss
            actual = 1 if home_score > away_score else 0
            ll = -(actual * np.log(max(p, 0.001)) + (1 - actual) * np.log(max(1 - p, 0.001)))
            log_losses.append(ll)

            status = "✅" if is_correct else "❌"
            game_label = f"{away} {away_score} - {home} {home_score}"
            print(f"  {game_label:<45s} {model_pick:>15} {actual_winner:>10} {status:>7}")

        print(f"\n  Accuracy: {correct}/{total_scored} ({correct/total_scored:.1%})")
        if log_losses:
            print(f"  Avg Log Loss: {np.mean(log_losses):.4f}")
    else:
        print("No completed tournament games yet. Run this cell after games finish!")
except Exception as e:
    print(f"No results table yet: {e}")
    print("Results will be tracked after Notebook 06 runs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook provides the complete bracket analysis with calibrated predictions.
# MAGIC
# MAGIC **Key improvement**: By blending model predictions with historical seed-based
# MAGIC win rates (α=0.4), we corrected the mid-major bias that caused our raw model
# MAGIC to pick 15 upsets in the first round (vs ~5-8 historically).
# MAGIC
# MAGIC **Re-run sections 9 and 10** after each round to track performance.
