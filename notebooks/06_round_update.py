# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 6: Round Update — Live Re-optimization
# MAGIC
# MAGIC **Run this after each tournament round completes.**
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Pulls latest tournament results from ESPN
# MAGIC 2. Updates Elo ratings with actual outcomes
# MAGIC 3. Recalculates features for surviving teams
# MAGIC 4. Re-weights the ensemble based on model performance
# MAGIC 5. Re-generates predictions for remaining games
# MAGIC 6. Tracks prediction accuracy per round
# MAGIC
# MAGIC ### Why Re-optimizing Works
# MAGIC
# MAGIC - Tournament games are **high-signal**: a team beating a higher seed tells us the committee underrated them
# MAGIC - Elo updates from tournament results adjust for "the committee got it wrong" scenarios
# MAGIC - Ensemble weights can shift based on which model predicted the completed round best
# MAGIC - Smaller remaining field = more precise predictions

# COMMAND ----------

# MAGIC %pip install requests xgboost scikit-learn matplotlib seaborn
# MAGIC %restart_python

# COMMAND ----------

import requests
import pandas as pd
import numpy as np
import json
import time
import traceback
import sys
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb

np.random.seed(42)

def log(msg):
    print(msg, flush=True)

def safe_int(v, default=0):
    if isinstance(v, dict): return default
    try: return int(v)
    except: return default

def safe_float(v, default=0.0):
    if isinstance(v, dict): return default
    try: return float(v)
    except: return default

ESPN_BASE = "site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh)"}

def espn_get(endpoint, params=None, retries=3):
    url = f"https://{ESPN_BASE}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HDR, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log(f"Failed: {url} — {e}")
                return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pull Latest Tournament Results

# COMMAND ----------

def parse_scoreboard_games(data, season=2026):
    """Parse ESPN scoreboard into game records with consistent schema."""
    games = []
    if not data or "events" not in data:
        return games

    for event in data.get("events", []):
        # Every game dict must have the exact same keys
        game = {
            "game_id": str(event.get("id", "")),
            "date": str(event.get("date", "")),
            "season": int(season),
            "name": str(event.get("name", "")),
            "short_name": str(event.get("shortName", "")),
            "neutral_site": False,
            "tournament_game": True,
            "status": "",
            "home_team_id": 0,
            "home_team_name": "",
            "home_score": 0,
            "home_winner": False,
            "home_seed": 0,
            "away_team_id": 0,
            "away_team_name": "",
            "away_score": 0,
            "away_winner": False,
            "away_seed": 0,
        }

        for comp in event.get("competitions", []):
            game["neutral_site"] = bool(comp.get("neutralSite", False))

            for c in comp.get("competitors", []):
                prefix = "home" if c.get("homeAway") == "home" else "away"
                team_data = c.get("team", {})
                game[f"{prefix}_team_id"] = safe_int(team_data.get("id", 0))
                game[f"{prefix}_team_name"] = str(team_data.get("displayName", ""))
                game[f"{prefix}_score"] = safe_int(c.get("score", 0))
                game[f"{prefix}_winner"] = bool(c.get("winner", False))

                seed_val = None
                if "curatedRank" in c:
                    seed_val = c["curatedRank"].get("current")
                if seed_val is None and "seed" in c:
                    seed_val = c.get("seed")
                game[f"{prefix}_seed"] = safe_int(seed_val) if seed_val else 0

            status = comp.get("status", {})
            game["status"] = str(status.get("type", {}).get("description", ""))

        games.append(game)

    return games

# Fetch all tournament games (March 17 through April 9)
log(">>> SECTION 1: Fetch tournament results from ESPN")
all_tourney_games = []
for month, days in [(3, range(17, 32)), (4, range(1, 10))]:
    for day in days:
        date_str = f"2026{month:02d}{day:02d}"
        data = espn_get("scoreboard", params={
            "dates": date_str, "groups": 50, "limit": 200, "seasontype": 3
        })
        games = parse_scoreboard_games(data)
        all_tourney_games.extend(games)
        time.sleep(0.05)

# Filter to completed games
completed = [g for g in all_tourney_games if g.get("status") == "Final"]
log(f"Total tournament games found: {len(all_tourney_games)}")
log(f"Completed games: {len(completed)}")

# Update the current_tourney_results table
if completed:
    pdf = pd.DataFrame(completed)
    # Explicit type casting to avoid Spark inference issues
    pdf["game_id"] = pdf["game_id"].astype(str)
    pdf["season"] = pdf["season"].astype(int)
    pdf["home_team_id"] = pdf["home_team_id"].astype(int)
    pdf["away_team_id"] = pdf["away_team_id"].astype(int)
    pdf["home_score"] = pdf["home_score"].astype(int)
    pdf["away_score"] = pdf["away_score"].astype(int)
    pdf["home_seed"] = pdf["home_seed"].astype(int)
    pdf["away_seed"] = pdf["away_seed"].astype(int)
    pdf["home_winner"] = pdf["home_winner"].astype(bool)
    pdf["away_winner"] = pdf["away_winner"].astype(bool)
    pdf["neutral_site"] = pdf["neutral_site"].astype(bool)
    pdf["tournament_game"] = pdf["tournament_game"].astype(bool)
    ct_df = spark.createDataFrame(pdf)
    ct_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("bracketology.raw.current_tourney_results")
    log("Updated bracketology.raw.current_tourney_results")
else:
    log("No completed games yet — skipping table update")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evaluate Model Performance on Completed Games

# COMMAND ----------

log(">>> SECTION 2: Evaluate Model Performance")
try:
    # Load model metadata and rebuild prediction capability
    metadata_raw = spark.table("bracketology.predictions.model_metadata").toPandas()
    metadata = {row["key"]: json.loads(row["value"]) for _, row in metadata_raw.iterrows()}

    FEATURE_COLS = metadata["feature_cols"]
    old_weights = np.array(metadata["ensemble_weights"])
    scaler_mean = np.array(metadata["scaler_mean"])
    scaler_scale = np.array(metadata["scaler_scale"])
    best_xgb_params = metadata["best_xgb_params"]
    for k in ["max_depth", "n_estimators", "min_child_weight"]:
        if k in best_xgb_params: best_xgb_params[k] = int(best_xgb_params[k])
    log(f"Metadata loaded: {len(FEATURE_COLS)} features, XGB params: {best_xgb_params}")
except Exception:
    traceback.print_exc()
    raise

# Load pairwise predictions from before this round
pairwise_df = spark.table("bracketology.predictions.pairwise_probabilities").toPandas()
prob_lookup = {}
for _, row in pairwise_df.iterrows():
    a, b = safe_int(row["team_a_id"]), safe_int(row["team_b_id"])
    prob_lookup[(a, b)] = row["p_team_a_wins"]
    prob_lookup[(b, a)] = row["p_team_b_wins"]

# Compare predictions to actual results
if completed:
    round_log = []
    for game in completed:
        home_id = game.get("home_team_id", 0)
        away_id = game.get("away_team_id", 0)

        # Find our predicted probability
        p_home = prob_lookup.get((home_id, away_id), 0.5)
        actual_home_win = 1 if game.get("home_winner", False) else 0

        round_log.append({
            "game": game.get("short_name", game.get("name", "")),
            "predicted_home_win_prob": p_home,
            "actual_home_win": actual_home_win,
            "correct": (p_home > 0.5) == (actual_home_win == 1),
            "log_loss_contribution": -(
                actual_home_win * np.log(max(p_home, 0.001)) +
                (1 - actual_home_win) * np.log(max(1 - p_home, 0.001))
            ),
        })

    round_df = pd.DataFrame(round_log)
    avg_ll = round_df["log_loss_contribution"].mean()
    accuracy = round_df["correct"].mean()

    log("=" * 65)
    log("ROUND PERFORMANCE REPORT")
    log("=" * 65)
    log(f"  Games evaluated: {len(round_df)}")
    log(f"  Average Log Loss: {avg_ll:.4f}")
    log(f"  Accuracy: {accuracy:.1%}")
    log(f"\nPer-game breakdown:")
    for _, g in round_df.iterrows():
        status = "CORRECT" if g["correct"] else "WRONG"
        log(f"  {g['game']:<40s} P={g['predicted_home_win_prob']:.2f} [{status}] LL={g['log_loss_contribution']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Update Elo Ratings with Tournament Results

# COMMAND ----------

log(">>> SECTION 3: Update Elo Ratings")
# Load current Elo ratings (latest per team from features)
features_df = spark.table("bracketology.features.team_season_features").toPandas()
current_elos = dict(zip(
    features_df[features_df["season"] == 2026]["team_id"].astype(int),
    features_df[features_df["season"] == 2026]["elo"]
))

# Update Elos with tournament results
K_TOURNEY = 32

for game in completed:
    home_id = safe_int(game.get("home_team_id", 0))
    away_id = safe_int(game.get("away_team_id", 0))
    home_score = game.get("home_score", 0)
    away_score = game.get("away_score", 0)

    if home_id == 0 or away_id == 0:
        continue

    home_elo = current_elos.get(home_id, 1500)
    away_elo = current_elos.get(away_id, 1500)

    # Expected outcome (neutral site)
    e_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))
    s_home = 1.0 if home_score > away_score else 0.0

    # Margin-of-victory adjustment
    margin = abs(home_score - away_score)
    winner_elo = home_elo if s_home == 1 else away_elo
    loser_elo = away_elo if s_home == 1 else home_elo
    elo_diff = winner_elo - loser_elo
    mov_mult = np.log(margin + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

    k = K_TOURNEY * mov_mult

    current_elos[home_id] = home_elo + k * (s_home - e_home)
    current_elos[away_id] = away_elo + k * ((1 - s_home) - (1 - e_home))

log(f"Updated Elo ratings for {len(current_elos)} teams")

# Show biggest Elo movers
if completed:
    original_elos = dict(zip(
        features_df[features_df["season"] == 2026]["team_id"].astype(int),
        features_df[features_df["season"] == 2026]["elo"]
    ))

    movers = []
    teams_raw = spark.table("bracketology.raw.teams").toPandas()
    for tid in current_elos:
        old = original_elos.get(tid, 1500)
        new = current_elos[tid]
        if abs(new - old) > 5:
            name = teams_raw[teams_raw["team_id"] == tid]["name"].values
            name = name[0] if len(name) > 0 else f"Team {tid}"
            movers.append({"team": name, "old_elo": old, "new_elo": new, "change": new - old})

    if movers:
        movers_df = pd.DataFrame(movers).sort_values("change", ascending=False)
        log("\nBiggest Elo Movers After Tournament Games:")
        for _, m in movers_df.head(10).iterrows():
            log(f"  {m['team']:<30s} {m['old_elo']:.0f} -> {m['new_elo']:.0f} ({m['change']:+.0f})")
        log("  ...")
        for _, m in movers_df.tail(10).iterrows():
            log(f"  {m['team']:<30s} {m['old_elo']:.0f} -> {m['new_elo']:.0f} ({m['change']:+.0f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Identify Surviving Teams

# COMMAND ----------

log(">>> SECTION 4: Identify Surviving Teams")
# Determine which teams are still alive
eliminated = set()

for game in completed:
    home_id = safe_int(game.get("home_team_id", 0))
    away_id = safe_int(game.get("away_team_id", 0))

    if game.get("home_winner", False):
        eliminated.add(away_id)
    elif game.get("away_winner", False):
        eliminated.add(home_id)

seeds_df = spark.table("bracketology.raw.tourney_seeds").toPandas()
seeds_2026 = seeds_df[seeds_df["season"] == 2026]
all_tourney_teams = set(seeds_2026["team_id"].astype(int).tolist())
surviving = all_tourney_teams - eliminated

log(f"Tournament field: {len(all_tourney_teams)} teams")
log(f"Eliminated: {len(eliminated)}")
log(f"Surviving: {len(surviving)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Re-weight Ensemble
# MAGIC
# MAGIC If one model significantly outperformed the others in completed rounds,
# MAGIC shift weight toward it.

# COMMAND ----------

log(">>> SECTION 5: Re-weight Ensemble")
try:
    # Rebuild models and evaluate each on completed tournament games
    matchup_df = spark.table("bracketology.features.matchup_features").toPandas()
    X_train = matchup_df[FEATURE_COLS].fillna(0).values.astype(float)
    y_train = matchup_df["team_a_won"].values.astype(int)
    log(f"Training data: {X_train.shape}")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    lr_model = LogisticRegression(C=0.1, max_iter=1000, penalty="l2")
    lr_model.fit(X_train_scaled, y_train)
    log("LR trained")

    xgb_model = xgb.XGBClassifier(**best_xgb_params, eval_metric="logloss", use_label_encoder=False, random_state=42)
    xgb_model.fit(X_train, y_train)
    log("XGB trained")

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate each model on completed tournament games
    if completed and len(completed) >= 3:
        model_lls = {"lr": [], "xgb": [], "rf": []}

        for game in completed:
            home_id = safe_int(game.get("home_team_id", 0))
            away_id = safe_int(game.get("away_team_id", 0))
            actual = 1 if game.get("home_winner", False) else 0

            # Build features
            feat_home = features_df[
                (features_df["team_id"] == home_id) & (features_df["season"] == 2026)
            ]
            feat_away = features_df[
                (features_df["team_id"] == away_id) & (features_df["season"] == 2026)
            ]

            if len(feat_home) == 0 or len(feat_away) == 0:
                continue

            fh = feat_home.iloc[0]
            fa = feat_away.iloc[0]

            feature_vals = []
            for f in FEATURE_COLS:
                if f == "elo_diff":
                    feature_vals.append(safe_float(fh.get("elo", 1500), 1500) - safe_float(fa.get("elo", 1500), 1500))
                elif f == "seed_diff":
                    feature_vals.append(safe_float(fa.get("seed", 8), 8) - safe_float(fh.get("seed", 8), 8))
                elif f.endswith("_diff"):
                    base = f.replace("_diff", "")
                    feature_vals.append(safe_float(fh.get(base, 0)) - safe_float(fa.get(base, 0)))
                elif f == "team_a_elo":
                    feature_vals.append(safe_float(fh.get("elo", 1500), 1500))
                elif f == "team_b_elo":
                    feature_vals.append(safe_float(fa.get("elo", 1500), 1500))
                elif f == "team_a_seed":
                    feature_vals.append(safe_float(fh.get("seed", 8), 8))
                elif f == "team_b_seed":
                    feature_vals.append(safe_float(fa.get("seed", 8), 8))
                else:
                    feature_vals.append(0)

            X_game = np.array([feature_vals])
            X_game = np.nan_to_num(X_game, nan=0.0, posinf=0.0, neginf=0.0)
            X_game_scaled = scaler.transform(X_game)

            p_lr = float(np.clip(lr_model.predict_proba(X_game_scaled)[0, 1], 0.01, 0.99))
            p_xgb = float(np.clip(xgb_model.predict_proba(X_game)[0, 1], 0.01, 0.99))
            p_rf = float(np.clip(rf_model.predict_proba(X_game)[0, 1], 0.01, 0.99))

            model_lls["lr"].append(-(actual * np.log(p_lr) + (1-actual) * np.log(1-p_lr)))
            model_lls["xgb"].append(-(actual * np.log(p_xgb) + (1-actual) * np.log(1-p_xgb)))
            model_lls["rf"].append(-(actual * np.log(p_rf) + (1-actual) * np.log(1-p_rf)))

        avg_lls = {k: np.mean(v) for k, v in model_lls.items()}
        log(f"Per-model tournament log loss:")
        log(f"  LR:  {avg_lls['lr']:.4f}")
        log(f"  XGB: {avg_lls['xgb']:.4f}")
        log(f"  RF:  {avg_lls['rf']:.4f}")

        # Re-weight: inverse log loss (better model gets more weight)
        inv_lls = {k: 1.0 / v for k, v in avg_lls.items()}
        total_inv = sum(inv_lls.values())
        new_weights = np.array([
            inv_lls["lr"] / total_inv,
            inv_lls["xgb"] / total_inv,
            inv_lls["rf"] / total_inv,
        ])

        # Blend with original weights (don't overreact to one round)
        blend_factor = 0.5  # 50% original, 50% updated
        updated_weights = blend_factor * old_weights + (1 - blend_factor) * new_weights
        updated_weights = updated_weights / updated_weights.sum()

        log(f"\nWeight update:")
        log(f"  Old weights: LR={old_weights[0]:.3f}, XGB={old_weights[1]:.3f}, RF={old_weights[2]:.3f}")
        log(f"  New weights: LR={updated_weights[0]:.3f}, XGB={updated_weights[1]:.3f}, RF={updated_weights[2]:.3f}")
    else:
        updated_weights = old_weights
        log("Not enough completed games to re-weight ensemble — keeping original weights")

except Exception:
    traceback.print_exc()
    raise

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Re-generate Predictions for Remaining Games

# COMMAND ----------

log(">>> SECTION 6: Re-generate Predictions")
try:
    # Update team features with new Elos
    updated_features = features_df[features_df["season"] == 2026].copy()
    elo_map = {safe_int(tid): safe_float(elo, 1500) for tid, elo in current_elos.items()}
    updated_features["elo"] = updated_features["team_id"].apply(lambda tid: elo_map.get(safe_int(tid), 1500))

    # Recalculate SOS with updated Elos
    # (simplified: just update the Elo-based metrics)

    # Generate new pairwise probabilities for surviving teams
    surviving_list = sorted(surviving)
    new_pairwise = []

    seeds_lookup = dict(zip(seeds_2026["team_id"].astype(int), seeds_2026["seed"].astype(int)))
    teams_raw = spark.table("bracketology.raw.teams").toPandas()
    name_lookup = dict(zip(teams_raw["team_id"].astype(int), teams_raw["name"]))

    for i in range(len(surviving_list)):
        for j in range(i + 1, len(surviving_list)):
            tid_a = surviving_list[i]
            tid_b = surviving_list[j]

            fa = updated_features[updated_features["team_id"] == tid_a]
            fb = updated_features[updated_features["team_id"] == tid_b]

            if len(fa) == 0 or len(fb) == 0:
                continue

            fa = fa.iloc[0]
            fb = fb.iloc[0]

            feature_vals = []
            for f in FEATURE_COLS:
                if f == "elo_diff":
                    feature_vals.append(safe_float(current_elos.get(tid_a, 1500), 1500) - safe_float(current_elos.get(tid_b, 1500), 1500))
                elif f == "seed_diff":
                    feature_vals.append(safe_float(fb.get("seed", 8), 8) - safe_float(fa.get("seed", 8), 8))
                elif f == "team_a_elo":
                    feature_vals.append(safe_float(current_elos.get(tid_a, 1500), 1500))
                elif f == "team_b_elo":
                    feature_vals.append(safe_float(current_elos.get(tid_b, 1500), 1500))
                elif f == "team_a_seed":
                    feature_vals.append(safe_float(fa.get("seed", 8), 8))
                elif f == "team_b_seed":
                    feature_vals.append(safe_float(fb.get("seed", 8), 8))
                elif f.endswith("_diff"):
                    base = f.replace("_diff", "")
                    feature_vals.append(safe_float(fa.get(base, 0)) - safe_float(fb.get(base, 0)))
                else:
                    feature_vals.append(0)

            X_pred = np.array([feature_vals])
            X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
            X_pred_scaled = scaler.transform(X_pred)

            p_lr = float(np.clip(lr_model.predict_proba(X_pred_scaled)[0, 1], 0.01, 0.99))
            p_xgb = float(np.clip(xgb_model.predict_proba(X_pred)[0, 1], 0.01, 0.99))
            p_rf = float(np.clip(rf_model.predict_proba(X_pred)[0, 1], 0.01, 0.99))

            p_ensemble = float(
                updated_weights[0] * p_lr +
                updated_weights[1] * p_xgb +
                updated_weights[2] * p_rf
            )
            p_ensemble = float(np.clip(p_ensemble, 0.01, 0.99))

            new_pairwise.append({
                "team_a_id": tid_a,
                "team_a_name": name_lookup.get(tid_a, f"Team {tid_a}"),
                "team_a_seed": safe_int(seeds_lookup.get(tid_a, 0)),
                "team_b_id": tid_b,
                "team_b_name": name_lookup.get(tid_b, f"Team {tid_b}"),
                "team_b_seed": safe_int(seeds_lookup.get(tid_b, 0)),
                "p_team_a_wins": p_ensemble,
                "p_team_b_wins": 1 - p_ensemble,
            })

    log(f"Generated {len(new_pairwise)} updated pairwise probabilities for {len(surviving)} surviving teams")
except Exception:
    traceback.print_exc()
    raise

# COMMAND ----------

log(">>> SECTION 6b: Save updated predictions")
if new_pairwise:
    new_pairwise_df = pd.DataFrame(new_pairwise)
    # Explicit type casting for Spark
    new_pairwise_df["team_a_id"] = new_pairwise_df["team_a_id"].astype(int)
    new_pairwise_df["team_b_id"] = new_pairwise_df["team_b_id"].astype(int)
    new_pairwise_df["team_a_seed"] = new_pairwise_df["team_a_seed"].astype(int)
    new_pairwise_df["team_b_seed"] = new_pairwise_df["team_b_seed"].astype(int)
    new_pairwise_df["p_team_a_wins"] = new_pairwise_df["p_team_a_wins"].astype(float)
    new_pairwise_df["p_team_b_wins"] = new_pairwise_df["p_team_b_wins"].astype(float)

    new_pairwise_sdf = spark.createDataFrame(new_pairwise_df)
    new_pairwise_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("bracketology.predictions.pairwise_probabilities")
    log(f"Saved {len(new_pairwise)} pairwise probabilities")

    # Save round update log
    round_update = {
        "update_timestamp": datetime.now().isoformat(),
        "games_completed": len(completed),
        "teams_eliminated": len(eliminated),
        "teams_remaining": len(surviving),
        "ensemble_weights": updated_weights.tolist(),
        "round_log_loss": float(avg_ll) if 'avg_ll' in dir() else None,
        "round_accuracy": float(accuracy) if 'accuracy' in dir() else None,
    }

    update_rows = [{"key": str(k), "value": json.dumps(v)} for k, v in round_update.items()]
    update_pdf = pd.DataFrame(update_rows)
    update_pdf["key"] = update_pdf["key"].astype(str)
    update_pdf["value"] = update_pdf["value"].astype(str)
    update_df = spark.createDataFrame(update_pdf)

    try:
        update_df.write.mode("append").saveAsTable("bracketology.predictions.round_updates")
    except Exception:
        update_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("bracketology.predictions.round_updates")

    log("Saved updated predictions and round log")
else:
    log("No pairwise predictions to save")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Updated Bracket Visualization

# COMMAND ----------

log(">>> SECTION 7: Monte Carlo simulation")
if new_pairwise:
    from collections import defaultdict

    n_sims = 10000
    champ_counts = defaultdict(int)

    # Reuse lookups from earlier sections
    updated_prob_lookup = {}
    for _, row in new_pairwise_df.iterrows():
        a, b = safe_int(row["team_a_id"]), safe_int(row["team_b_id"])
        updated_prob_lookup[(a, b)] = row["p_team_a_wins"]
        updated_prob_lookup[(b, a)] = row["p_team_b_wins"]

    surviving_sorted = sorted(surviving, key=lambda t: seeds_lookup.get(t, 16))
    log(f"Running {n_sims} simulations with {len(surviving_sorted)} teams...")

    for sim in range(n_sims):
        remaining = list(surviving_sorted)
        np.random.shuffle(remaining)

        while len(remaining) > 1:
            next_round = []
            for i in range(0, len(remaining) - 1, 2):
                t1, t2 = remaining[i], remaining[i+1]
                p = updated_prob_lookup.get((t1, t2), 0.5)
                winner = t1 if np.random.random() < p else t2
                next_round.append(winner)
            if len(remaining) % 2 == 1:
                next_round.append(remaining[-1])
            remaining = next_round

        if remaining:
            champ_counts[remaining[0]] += 1

    log("=" * 65)
    log("UPDATED CHAMPIONSHIP ODDS (Post-Round Update)")
    log("=" * 65)

    sorted_contenders = sorted(champ_counts.items(), key=lambda x: x[1], reverse=True)
    for tid, count in sorted_contenders[:15]:
        prob = count / n_sims
        seed = seeds_lookup.get(tid, 0)
        name = name_lookup.get(tid, f"Team {tid}")
        bar = "#" * int(prob * 100)
        log(f"  [{seed:2d}] {name:<30s} {prob:6.1%} {bar}")

    total_prob = sum(c / n_sims for _, c in sorted_contenders)
    log(f"\n  Total probability: {total_prob:.4f}")
else:
    log("No updated predictions — skipping Monte Carlo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC After this update:
# MAGIC - Elo ratings reflect tournament performance
# MAGIC - Ensemble weights adjusted based on round accuracy
# MAGIC - Predictions regenerated for surviving teams only
# MAGIC - Performance tracked in `bracketology.predictions.round_updates`
# MAGIC
# MAGIC **Run this notebook again after each subsequent round.**
