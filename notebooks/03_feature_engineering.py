# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 3: Feature Engineering
# MAGIC
# MAGIC **Goal**: Transform raw game data into model-ready features stored in `bracketology.features.*`
# MAGIC
# MAGIC This notebook computes three tables:
# MAGIC - `elo_ratings` — running Elo for every team across all seasons
# MAGIC - `team_season_features` — per-team-season stat profile
# MAGIC - `matchup_features` — pairwise features for training and prediction
# MAGIC
# MAGIC > **Why not just use raw stats?** Raw stats are contaminated by pace, opponent quality,
# MAGIC > and schedule strength. Feature engineering removes these confounds so the model sees
# MAGIC > true team ability.

# COMMAND ----------

# MAGIC %pip install pandas numpy
# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Elo Ratings
# MAGIC
# MAGIC ### How Elo Works
# MAGIC
# MAGIC Elo is a **dynamic rating system** originally designed for chess. For each game:
# MAGIC
# MAGIC 1. **Expected outcome**: `E_A = 1 / (1 + 10^((R_B - R_A) / 400))`
# MAGIC 2. **Update**: `R_A_new = R_A + K * (S_A - E_A)`
# MAGIC
# MAGIC Where:
# MAGIC - `R_A, R_B` = current ratings of teams A and B
# MAGIC - `S_A` = actual outcome (1 = win, 0 = loss)
# MAGIC - `K` = update magnitude (higher = more reactive)
# MAGIC - `E_A` = expected probability of A winning
# MAGIC
# MAGIC ### Our Enhancements
# MAGIC
# MAGIC | Enhancement | Why |
# MAGIC |-------------|-----|
# MAGIC | **Margin-of-victory adjustment** | A 30-point win is more informative than a 1-point win |
# MAGIC | **Home-court advantage** | Home teams win ~60% in college basketball |
# MAGIC | **Season-to-season regression** | Teams change year to year — regress 1/3 toward mean |
# MAGIC | **K-factor tuning** | K=20 for regular season, K=32 for tournament (higher stakes, more signal) |

# COMMAND ----------

def compute_elo_ratings(games_df, initial_elo=1500, k_regular=20, k_tourney=32,
                        home_advantage=100, mov_multiplier=True, season_regression=0.33):
    """
    Compute Elo ratings game-by-game for all teams across multiple seasons.

    Returns: dict of {team_id: current_elo} and list of per-game Elo snapshots.
    """
    elos = {}  # team_id -> current elo
    elo_history = []  # list of per-game records

    # Sort games chronologically
    games = games_df.sort_values("date").reset_index(drop=True)

    current_season = None

    for _, game in games.iterrows():
        season = game.get("season", 2026)

        # Season-to-season regression
        if current_season is not None and season != current_season:
            mean_elo = np.mean(list(elos.values())) if elos else initial_elo
            for tid in elos:
                elos[tid] = elos[tid] * (1 - season_regression) + mean_elo * season_regression
            current_season = season
        elif current_season is None:
            current_season = season

        # Get team IDs
        home_id = game.get("home_team_id")
        away_id = game.get("away_team_id")

        if pd.isna(home_id) or pd.isna(away_id):
            continue

        home_id = int(home_id)
        away_id = int(away_id)

        # Initialize if new team
        if home_id not in elos:
            elos[home_id] = initial_elo
        if away_id not in elos:
            elos[away_id] = initial_elo

        # Current ratings
        home_elo = elos[home_id]
        away_elo = elos[away_id]

        # Adjust for home court (unless neutral site)
        neutral = game.get("neutral_site", False)
        hca = 0 if neutral else home_advantage

        # Expected outcomes
        e_home = 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + hca)) / 400))
        e_away = 1.0 - e_home

        # Actual outcomes
        home_score = game.get("home_score", 0) or 0
        away_score = game.get("away_score", 0) or 0

        if home_score == 0 and away_score == 0:
            continue  # Skip games without scores

        s_home = 1.0 if home_score > away_score else 0.0
        s_away = 1.0 - s_home

        # K-factor
        is_tourney = game.get("tournament_game", False)
        k = k_tourney if is_tourney else k_regular

        # Margin-of-victory adjustment
        if mov_multiplier:
            margin = abs(home_score - away_score)
            # Logarithmic MOV multiplier (diminishing returns for blowouts)
            # FiveThirtyEight-style: multiplier = ln(margin + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
            winner_elo = home_elo if s_home == 1 else away_elo
            loser_elo = away_elo if s_home == 1 else home_elo
            elo_diff = winner_elo - loser_elo
            mov_mult = np.log(margin + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
            k = k * mov_mult

        # Update ratings
        elos[home_id] = home_elo + k * (s_home - e_home)
        elos[away_id] = away_elo + k * (s_away - e_away)

        # Record
        elo_history.append({
            "game_id": game.get("game_id", ""),
            "date": game.get("date", ""),
            "season": season,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_elo_before": home_elo,
            "away_elo_before": away_elo,
            "home_elo_after": elos[home_id],
            "away_elo_after": elos[away_id],
            "home_score": home_score,
            "away_score": away_score,
            "home_win_prob": e_home,
            "actual_home_win": s_home,
            "tournament_game": is_tourney,
        })

    return elos, elo_history

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute Elo Across All Historical + Current Data

# COMMAND ----------

# Load all game data, combine, and sort chronologically
historical_df = spark.table("bracketology.raw.historical_seasons").toPandas()
tourney_df = spark.table("bracketology.raw.historical_tourney").toPandas()
current_df = spark.table("bracketology.raw.regular_season_games").toPandas()

# Standardize columns for historical schedule data (team-centric -> game-centric)
if "opponent_id" in historical_df.columns and "home_team_id" not in historical_df.columns:
    # Convert team-schedule format to game format
    hist_games = []
    for _, row in historical_df.iterrows():
        if row.get("home_away") == "home":
            hist_games.append({
                "game_id": row.get("game_id", ""),
                "date": row.get("date", ""),
                "season": row.get("season", 0),
                "home_team_id": row.get("team_id", 0),
                "away_team_id": row.get("opponent_id", 0),
                "home_score": row.get("score", 0),
                "away_score": row.get("opponent_score", 0),
                "neutral_site": row.get("neutral_site", False),
                "tournament_game": row.get("season_type", 0) == 3,
            })
        else:
            hist_games.append({
                "game_id": row.get("game_id", ""),
                "date": row.get("date", ""),
                "season": row.get("season", 0),
                "home_team_id": row.get("opponent_id", 0),
                "away_team_id": row.get("team_id", 0),
                "home_score": row.get("opponent_score", 0),
                "away_score": row.get("score", 0),
                "neutral_site": row.get("neutral_site", False),
                "tournament_game": row.get("season_type", 0) == 3,
            })
    historical_df = pd.DataFrame(hist_games)

# Mark tournament games
if "tournament_game" not in tourney_df.columns:
    tourney_df["tournament_game"] = True
if "tournament_game" not in current_df.columns:
    current_df["tournament_game"] = False
if "tournament_game" not in historical_df.columns:
    historical_df["tournament_game"] = False

# Combine all games
common_cols = ["game_id", "date", "season", "home_team_id", "away_team_id",
               "home_score", "away_score", "neutral_site", "tournament_game"]

def safe_select(df, cols):
    result = {}
    for c in cols:
        if c in df.columns:
            result[c] = df[c]
        else:
            result[c] = None
    return pd.DataFrame(result)

all_games = pd.concat([
    safe_select(historical_df, common_cols),
    safe_select(tourney_df, common_cols),
    safe_select(current_df, common_cols),
], ignore_index=True)

# Deduplicate by game_id
all_games = all_games.drop_duplicates(subset=["game_id"], keep="first")
all_games = all_games.sort_values("date").reset_index(drop=True)

# Filter to completed games with scores
all_games = all_games[
    (all_games["home_score"].fillna(0) > 0) | (all_games["away_score"].fillna(0) > 0)
]

print(f"Total unique games for Elo computation: {len(all_games):,}")
print(f"Seasons: {sorted(all_games['season'].unique())}")

# COMMAND ----------

# Compute Elo ratings
final_elos, elo_history = compute_elo_ratings(all_games)

print(f"Elo history records: {len(elo_history):,}")
print(f"Teams with ratings: {len(final_elos)}")
print(f"Mean Elo: {np.mean(list(final_elos.values())):.0f}")
print(f"Max Elo: {max(final_elos.values()):.0f}")
print(f"Min Elo: {min(final_elos.values()):.0f}")

# COMMAND ----------

# Write Elo history to Delta
elo_history_df = pd.DataFrame(elo_history)
elo_sdf = spark.createDataFrame(elo_history_df)
elo_sdf.write.mode("overwrite").saveAsTable("bracketology.features.elo_ratings")

count = spark.table("bracketology.features.elo_ratings").count()
print(f"bracketology.features.elo_ratings: {count:,} rows")

# Show top 25 teams by current Elo
teams_df = spark.table("bracketology.raw.teams").toPandas()
top_elos = sorted(final_elos.items(), key=lambda x: x[1], reverse=True)[:25]
for rank, (tid, elo) in enumerate(top_elos, 1):
    name = teams_df[teams_df["team_id"] == tid]["name"].values
    name = name[0] if len(name) > 0 else f"Team {tid}"
    print(f"  {rank:2d}. {name:<30s} Elo: {elo:.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Team Season Features
# MAGIC
# MAGIC For each team-season, we compute a comprehensive feature profile.
# MAGIC These features are designed to be **pace-independent** and **opponent-adjusted** where possible.

# COMMAND ----------

def compute_team_season_features(games_df, elo_dict, seeds_df, season=2026):
    """
    Compute per-team features for a given season.

    Features computed:
    - Record (wins, losses, win%)
    - Elo rating (final)
    - Scoring averages (points for/against, margin)
    - Close game record (decided by ≤5 points)
    - Away/neutral record
    - Last-10 game Elo delta (momentum)
    - Consistency (std dev of scoring margin)
    - Turnover-related proxies
    - Free throw rate proxies
    """
    season_games = games_df[games_df["season"] == season].copy()

    teams = set(season_games["home_team_id"].dropna().astype(int).tolist() +
                season_games["away_team_id"].dropna().astype(int).tolist())

    features_list = []

    for team_id in teams:
        team_id = int(team_id)

        # Get all games for this team
        home_games = season_games[season_games["home_team_id"] == team_id]
        away_games = season_games[season_games["away_team_id"] == team_id]

        # Calculate basic record
        home_wins = ((home_games["home_score"] > home_games["away_score"])).sum()
        away_wins = ((away_games["away_score"] > away_games["home_score"])).sum()
        total_wins = home_wins + away_wins
        total_games = len(home_games) + len(away_games)
        total_losses = total_games - total_wins

        if total_games == 0:
            continue

        # Scoring
        pts_for_home = home_games["home_score"].sum()
        pts_for_away = away_games["away_score"].sum()
        pts_against_home = home_games["away_score"].sum()
        pts_against_away = away_games["home_score"].sum()

        total_pts_for = pts_for_home + pts_for_away
        total_pts_against = pts_against_home + pts_against_away

        avg_pts_for = total_pts_for / total_games
        avg_pts_against = total_pts_against / total_games
        avg_margin = avg_pts_for - avg_pts_against

        # Close games (within 5 points)
        home_margins = (home_games["home_score"] - home_games["away_score"])
        away_margins = (away_games["away_score"] - away_games["home_score"])
        all_margins = pd.concat([home_margins, away_margins])

        close_games = all_margins.abs() <= 5
        close_game_count = close_games.sum()
        close_game_wins = ((all_margins > 0) & close_games).sum()
        close_game_win_pct = close_game_wins / close_game_count if close_game_count > 0 else 0.5

        # Away + neutral record
        neutral_home = home_games[home_games.get("neutral_site", pd.Series(False, index=home_games.index)) == True]
        away_total = len(away_games) + len(neutral_home)
        away_wins_total = away_wins + ((neutral_home["home_score"] > neutral_home["away_score"])).sum() if len(neutral_home) > 0 else away_wins
        away_win_pct = away_wins_total / away_total if away_total > 0 else 0.5

        # Consistency (std dev of scoring margins)
        consistency = all_margins.std() if len(all_margins) > 1 else 15.0

        # Elo
        team_elo = elo_dict.get(team_id, 1500)

        # Last-10 Elo delta (momentum)
        team_elo_history = elo_history_df[
            ((elo_history_df["home_team_id"] == team_id) | (elo_history_df["away_team_id"] == team_id)) &
            (elo_history_df["season"] == season)
        ].sort_values("date")

        if len(team_elo_history) >= 10:
            last_10 = team_elo_history.tail(10)
            elo_10_ago = last_10.iloc[0]
            elo_now = last_10.iloc[-1]

            if elo_10_ago["home_team_id"] == team_id:
                elo_start = elo_10_ago["home_elo_before"]
            else:
                elo_start = elo_10_ago["away_elo_before"]

            if elo_now["home_team_id"] == team_id:
                elo_end = elo_now["home_elo_after"]
            else:
                elo_end = elo_now["away_elo_after"]

            last_10_elo_delta = elo_end - elo_start
        else:
            last_10_elo_delta = 0

        # Strength of schedule (average opponent Elo)
        opponent_ids = (
            home_games["away_team_id"].dropna().astype(int).tolist() +
            away_games["home_team_id"].dropna().astype(int).tolist()
        )
        opp_elos = [elo_dict.get(int(oid), 1500) for oid in opponent_ids]
        sos = np.mean(opp_elos) if opp_elos else 1500

        # Seed
        team_seeds = seeds_df[(seeds_df["team_id"] == team_id) & (seeds_df["season"] == season)]
        seed = int(team_seeds["seed"].iloc[0]) if len(team_seeds) > 0 else 0

        features_list.append({
            "team_id": team_id,
            "season": season,
            "wins": int(total_wins),
            "losses": int(total_losses),
            "win_pct": total_wins / total_games,
            "elo": team_elo,
            "avg_pts_for": avg_pts_for,
            "avg_pts_against": avg_pts_against,
            "avg_margin": avg_margin,
            "close_game_win_pct": close_game_win_pct,
            "close_games_played": int(close_game_count),
            "away_neutral_win_pct": away_win_pct,
            "consistency": consistency,
            "last_10_elo_delta": last_10_elo_delta,
            "sos": sos,
            "seed": seed,
            "total_games": int(total_games),
        })

    return pd.DataFrame(features_list)

# COMMAND ----------

# Compute features for current season + all historical seasons
seeds_df = spark.table("bracketology.raw.tourney_seeds").toPandas()

all_features = []
seasons_to_process = sorted(all_games["season"].unique())

for season in seasons_to_process:
    print(f"Computing features for season {season}...")
    season_features = compute_team_season_features(all_games, final_elos, seeds_df, season)
    all_features.append(season_features)
    print(f"  -> {len(season_features)} teams")

features_combined = pd.concat(all_features, ignore_index=True)
print(f"\nTotal team-season feature rows: {len(features_combined):,}")

# COMMAND ----------

# Enrich with ESPN team stats for current season
team_stats = spark.table("bracketology.raw.team_season_stats").toPandas()

if len(team_stats) > 0:
    # Select the most useful stat columns (avoid duplicates with our computed features)
    stat_cols_to_keep = ["team_id"]
    for col in team_stats.columns:
        if col not in ("team_id", "season") and team_stats[col].dtype in [np.float64, np.int64, float, int]:
            stat_cols_to_keep.append(col)

    # Merge ESPN stats with our features for current season only
    current_features = features_combined[features_combined["season"] == 2026].copy()
    enriched = current_features.merge(
        team_stats[stat_cols_to_keep],
        on="team_id",
        how="left"
    )

    # Replace current season features with enriched version
    features_combined = pd.concat([
        features_combined[features_combined["season"] != 2026],
        enriched
    ], ignore_index=True)

    print(f"Enriched 2026 features with {len(stat_cols_to_keep)-1} ESPN stat columns")

# COMMAND ----------

# Write team season features
features_sdf = spark.createDataFrame(features_combined.fillna(0))
features_sdf.write.mode("overwrite").saveAsTable("bracketology.features.team_season_features")

count = spark.table("bracketology.features.team_season_features").count()
print(f"bracketology.features.team_season_features: {count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Matchup Features
# MAGIC
# MAGIC For model training, we need **pairwise features** for each historical tournament game:
# MAGIC - `team_a` features - `team_b` features = difference features
# MAGIC - The label is whether team_a won
# MAGIC
# MAGIC We also generate features for all possible 2026 tournament matchups (for prediction).

# COMMAND ----------

def create_matchup_features(tourney_games_df, features_df, seeds_df):
    """
    Create pairwise matchup features for tournament games.
    Each row: (team_a, team_b, feature_diffs, team_a_won)
    """
    matchups = []

    for _, game in tourney_games_df.iterrows():
        season = game.get("season", 0)
        home_id = game.get("home_team_id")
        away_id = game.get("away_team_id")

        if pd.isna(home_id) or pd.isna(away_id):
            continue

        home_id = int(home_id)
        away_id = int(away_id)

        # Get features for both teams in this season
        home_feats = features_df[(features_df["team_id"] == home_id) & (features_df["season"] == season)]
        away_feats = features_df[(features_df["team_id"] == away_id) & (features_df["season"] == season)]

        if len(home_feats) == 0 or len(away_feats) == 0:
            continue

        home_feats = home_feats.iloc[0]
        away_feats = away_feats.iloc[0]

        # Compute difference features (home - away)
        matchup = {
            "game_id": game.get("game_id", ""),
            "season": season,
            "team_a_id": home_id,
            "team_b_id": away_id,
            "elo_diff": home_feats.get("elo", 1500) - away_feats.get("elo", 1500),
            "seed_diff": away_feats.get("seed", 8) - home_feats.get("seed", 8),  # Lower seed is better
            "win_pct_diff": home_feats.get("win_pct", 0.5) - away_feats.get("win_pct", 0.5),
            "avg_margin_diff": home_feats.get("avg_margin", 0) - away_feats.get("avg_margin", 0),
            "sos_diff": home_feats.get("sos", 1500) - away_feats.get("sos", 1500),
            "close_game_diff": home_feats.get("close_game_win_pct", 0.5) - away_feats.get("close_game_win_pct", 0.5),
            "away_win_diff": home_feats.get("away_neutral_win_pct", 0.5) - away_feats.get("away_neutral_win_pct", 0.5),
            "consistency_diff": away_feats.get("consistency", 15) - home_feats.get("consistency", 15),  # Lower is better
            "momentum_diff": home_feats.get("last_10_elo_delta", 0) - away_feats.get("last_10_elo_delta", 0),
            # Individual team features (for tree models that can use non-linear splits)
            "team_a_elo": home_feats.get("elo", 1500),
            "team_b_elo": away_feats.get("elo", 1500),
            "team_a_seed": home_feats.get("seed", 8),
            "team_b_seed": away_feats.get("seed", 8),
        }

        # Outcome
        home_score = game.get("home_score", 0) or 0
        away_score = game.get("away_score", 0) or 0
        matchup["team_a_won"] = 1 if home_score > away_score else 0
        matchup["margin"] = home_score - away_score

        matchups.append(matchup)

    return pd.DataFrame(matchups)

# COMMAND ----------

# Build training matchup features from historical tournament games
tourney_games = spark.table("bracketology.raw.historical_tourney").toPandas()

# Filter to completed games
tourney_games = tourney_games[tourney_games["status"] == "Final"] if "status" in tourney_games.columns else tourney_games

matchup_features = create_matchup_features(tourney_games, features_combined, seeds_df)
print(f"Historical tournament matchup features: {len(matchup_features)} games")

# Also add current tournament games if any
try:
    current_tourney = spark.table("bracketology.raw.current_tourney_results").toPandas()
    if len(current_tourney) > 0:
        current_tourney_completed = current_tourney[current_tourney.get("status", "") == "Final"] if "status" in current_tourney.columns else current_tourney
        if len(current_tourney_completed) > 0:
            current_matchups = create_matchup_features(current_tourney_completed, features_combined, seeds_df)
            matchup_features = pd.concat([matchup_features, current_matchups], ignore_index=True)
            print(f"Added {len(current_matchups)} current tournament matchups")
except Exception as e:
    print(f"No current tournament results to add: {e}")

# COMMAND ----------

# Write matchup features
if len(matchup_features) > 0:
    matchup_sdf = spark.createDataFrame(matchup_features)
    matchup_sdf.write.mode("overwrite").saveAsTable("bracketology.features.matchup_features")

    count = spark.table("bracketology.features.matchup_features").count()
    print(f"bracketology.features.matchup_features: {count:,} rows")

    # Show feature distributions
    print("\nFeature statistics:")
    print(matchup_features.describe().round(2).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

tables = [
    "bracketology.features.elo_ratings",
    "bracketology.features.team_season_features",
    "bracketology.features.matchup_features",
]

print("=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)
for table in tables:
    try:
        count = spark.table(table).count()
        cols = len(spark.table(table).columns)
        print(f"  {table}: {count:,} rows, {cols} columns")
    except Exception as e:
        print(f"  {table}: ERROR — {e}")

# Verify Elo sanity
elo_df = spark.table("bracketology.features.elo_ratings")
print(f"\nElo sanity check:")
print(f"  Records: {elo_df.count():,}")

# Feature completeness for tournament teams
tourney_features = spark.sql("""
    SELECT COUNT(*) as teams, AVG(elo) as avg_elo, AVG(sos) as avg_sos
    FROM bracketology.features.team_season_features
    WHERE season = 2026 AND seed > 0
""")
tourney_features.display()
print("=" * 60)
