# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 5: Tournament Predictions

# COMMAND ----------

# MAGIC %pip install xgboost scikit-learn pandas numpy matplotlib seaborn
# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import json
import traceback
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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

# Load metadata
try:
    meta_raw = spark.table("bracketology.predictions.model_metadata").toPandas()
    meta = {r["key"]: json.loads(r["value"]) for _, r in meta_raw.iterrows()}
    FEATURE_COLS = meta["feature_cols"]
    ew = np.array(meta["ensemble_weights"], dtype=float)
    sm = np.array(meta["scaler_mean"], dtype=float)
    ss = np.array(meta["scaler_scale"], dtype=float)
    xp = meta["best_xgb_params"]
    for k in ["max_depth", "n_estimators", "min_child_weight"]:
        if k in xp: xp[k] = int(xp[k])
    print(f"Metadata loaded: {len(FEATURE_COLS)} features")
except Exception:
    traceback.print_exc()
    raise

# COMMAND ----------

# Train models
try:
    mdf = spark.table("bracketology.features.matchup_features").toPandas()
    X = mdf[FEATURE_COLS].fillna(0).values.astype(float)
    y = mdf["team_a_won"].values.astype(int)
    print(f"Training data: {X.shape}")

    scaler = StandardScaler()
    scaler.fit(X)  # Fit fresh instead of loading params
    Xs = scaler.transform(X)

    lr = LogisticRegression(C=0.1, max_iter=1000, penalty="l2")
    lr.fit(Xs, y)

    xgb_m = xgb.XGBClassifier(**xp, eval_metric="logloss", use_label_encoder=False, random_state=42)
    xgb_m.fit(X, y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, max_features="sqrt", random_state=42)
    rf.fit(X, y)
    print("Models trained")
except Exception:
    traceback.print_exc()
    raise

# COMMAND ----------

# Load tournament field
try:
    seeds = spark.table("bracketology.raw.tourney_seeds").filter("season = 2026").toPandas()
    feats = spark.table("bracketology.features.team_season_features").filter("season = 2026").toPandas()

    # Drop seed from features to avoid duplicate column on merge
    if "seed" in feats.columns:
        feats = feats.drop(columns=["seed"])

    field = seeds.merge(feats, on=["team_id", "season"], how="left")
    field = field.sort_values("seed").reset_index(drop=True)
    # Kill ALL NaN — sklearn crashes on NaN input
    for c in field.columns:
        if field[c].dtype in ["float64", "float32"]:
            field[c] = field[c].fillna(0.0)
        elif field[c].dtype in ["int64", "int32"]:
            field[c] = field[c].fillna(0)

    tf = {}
    for _, r in field.iterrows():
        d = r.to_dict()
        # Ensure numeric types
        d["team_id"] = safe_int(d.get("team_id",0))
        d["seed"] = safe_int(d.get("seed",0))
        d["elo"] = safe_float(d.get("elo",1500), 1500)
        tf[d["team_id"]] = d

    print(f"Tournament field: {len(field)} teams")
    print(f"Sample team keys: {list(list(tf.values())[0].keys())[:10]}")
except Exception:
    traceback.print_exc()
    raise

# COMMAND ----------

# Generate pairwise probabilities
try:
    tids = [int(t) for t in field["team_id"].tolist()]

    def pred(a_id, b_id):
        fa, fb = tf[a_id], tf[b_id]
        fv = []
        for f in FEATURE_COLS:
            if f == "elo_diff": fv.append(safe_float(fa.get("elo",1500),1500) - safe_float(fb.get("elo",1500),1500))
            elif f == "seed_diff": fv.append(safe_float(fb.get("seed",8),8) - safe_float(fa.get("seed",8),8))
            elif f == "team_a_elo": fv.append(safe_float(fa.get("elo",1500),1500))
            elif f == "team_b_elo": fv.append(safe_float(fb.get("elo",1500),1500))
            elif f == "team_a_seed": fv.append(safe_float(fa.get("seed",8),8))
            elif f == "team_b_seed": fv.append(safe_float(fb.get("seed",8),8))
            elif f.endswith("_diff"):
                base = f.replace("_diff","")
                key_map = {"win_pct":"win_pct","avg_margin":"avg_margin","sos":"sos",
                           "close_game":"close_game_win_pct","away_win":"away_neutral_win_pct",
                           "consistency":"consistency","momentum":"last_10_elo_delta"}
                actual = key_map.get(base, base)
                fv.append(safe_float(fa.get(actual,0)) - safe_float(fb.get(actual,0)))
            else: fv.append(0.0)
        Xp = np.nan_to_num(np.array([fv], dtype=float))
        Xps = scaler.transform(Xp)
        p1 = float(lr.predict_proba(Xps)[0,1])
        p2 = float(xgb_m.predict_proba(Xp)[0,1])
        p3 = float(rf.predict_proba(Xp)[0,1])
        return float(np.clip(ew[0]*p1 + ew[1]*p2 + ew[2]*p3, 0.01, 0.99))

    # Test one prediction
    if len(tids) >= 2:
        test_p = pred(tids[0], tids[1])
        print(f"Test prediction: {tf[tids[0]].get('team_name','')} vs {tf[tids[1]].get('team_name','')}: {test_p:.3f}")

    pairs = []
    for i in range(len(tids)):
        for j in range(i+1, len(tids)):
            p = pred(tids[i], tids[j])
            pairs.append({
                "team_a_id": tids[i], "team_a_name": str(tf[tids[i]].get("team_name","")),
                "team_a_seed": safe_int(tf[tids[i]].get("seed",0)),
                "team_b_id": tids[j], "team_b_name": str(tf[tids[j]].get("team_name","")),
                "team_b_seed": int(tf[tids[j]].get("seed",0)),
                "p_team_a_wins": p, "p_team_b_wins": 1.0-p,
            })

    print(f"Pairwise matchups: {len(pairs)}")
    pdf = pd.DataFrame(pairs)
    spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable("bracketology.predictions.pairwise_probabilities")
    print(f"  Saved: {spark.table('bracketology.predictions.pairwise_probabilities').count()}")
except Exception:
    traceback.print_exc()
    raise

# COMMAND ----------

# Monte Carlo simulation
try:
    # Probability lookup
    pl = {}
    for _, r in pdf.iterrows():
        a, b = int(r["team_a_id"]), int(r["team_b_id"])
        pl[(a,b)] = float(r["p_team_a_wins"])
        pl[(b,a)] = float(r["p_team_b_wins"])

    # Build bracket by seed
    by_seed = {}
    for _, r in field.iterrows():
        s = int(r["seed"])
        by_seed.setdefault(s, []).append(safe_int(r["team_id"]))

    SEED_ORDER = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
    initial = []
    used = set()
    for s1, s2 in SEED_ORDER:
        for a, b in zip(by_seed.get(s1,[]), by_seed.get(s2,[])):
            initial.append((a,b))
            used.add(a); used.add(b)

    # Leftover
    for tid in tids:
        if tid not in used:
            # Pair with another unused team
            pass
    leftover = [t for t in tids if t not in used]
    for i in range(0, len(leftover)-1, 2):
        initial.append((leftover[i], leftover[i+1]))

    print(f"Initial matchups: {len(initial)}")

    N = 10000
    rounds_list = ["R64","R32","Sweet16","Elite8","Final4","Championship"]
    adv = {t: {rd:0 for rd in rounds_list+["Champion"]} for t in tids}

    for sim in range(N):
        bracket = list(initial)
        last_winners = []
        for ri, rd in enumerate(rounds_list):
            winners = []
            for a, b in bracket:
                p = pl.get((a,b), 0.5)
                w = a if np.random.random() < p else b
                winners.append(w)
                adv[w][rd] += 1
            last_winners = winners
            bracket = [(winners[i], winners[i+1]) for i in range(0, len(winners)-1, 2)]
            if len(bracket) == 0:
                break
        if last_winners:
            adv[last_winners[-1]]["Champion"] += 1

    # Build results
    sim_rows = []
    for _, r in field.iterrows():
        t = safe_int(r["team_id"])
        row = {"team_id": t, "team_name": str(r.get("team_name","")), "seed": safe_int(r.get("seed",0))}
        for rd in rounds_list+["Champion"]:
            row[f"p_{rd.lower()}"] = adv[t][rd] / N
        sim_rows.append(row)

    sim_df = pd.DataFrame(sim_rows)
    spark.createDataFrame(sim_df).write.mode("overwrite").saveAsTable("bracketology.predictions.bracket_simulation")
    print(f"Simulation saved: {len(sim_df)} teams")
except Exception:
    traceback.print_exc()
    raise

# COMMAND ----------

# Results
try:
    top = sim_df.sort_values("p_champion", ascending=False).head(25)
    print("=" * 85)
    print("2026 NCAA TOURNAMENT — CHAMPIONSHIP ODDS (10,000 Simulations)")
    print("=" * 85)
    print(f"{'Seed':>4}  {'Team':<30s} {'R32':>6} {'S16':>6} {'E8':>6} {'F4':>6} {'Final':>6} {'Champ':>7}")
    print("-" * 85)
    for _, t in top.iterrows():
        print(f"[{int(t['seed']):2d}]  {str(t['team_name']):<30s} "
              f"{t.get('p_r32',0):.1%}  {t.get('p_sweet16',0):.1%}  {t.get('p_elite8',0):.1%}  "
              f"{t.get('p_final4',0):.1%}  {t.get('p_championship',0):.1%}  {t.get('p_champion',0):.1%}")
    print(f"\nTotal champion prob: {sim_df['p_champion'].sum():.4f}")
except Exception:
    traceback.print_exc()

# COMMAND ----------

# Upset Watch
try:
    upsets = []
    for a, b in initial:
        sa, sb = int(tf[a].get("seed",8)), int(tf[b].get("seed",8))
        if sa == sb: continue
        fav, dog = (a,b) if sa < sb else (b,a)
        sf, sd = min(sa,sb), max(sa,sb)
        p_up = pl.get((dog, fav), 0.5)
        if p_up > 0.25 and sd - sf >= 3:
            upsets.append({"fav": f"[{sf}] {tf[fav].get('team_name','')}", "dog": f"[{sd}] {tf[dog].get('team_name','')}",
                          "prob": p_up, "diff": sd-sf})

    upsets.sort(key=lambda x: x["prob"], reverse=True)
    if upsets:
        print("\n" + "=" * 75)
        print("UPSET WATCH — First Round Underdogs >25%")
        print("=" * 75)
        for u in upsets[:15]:
            bar = "#" * int(u["prob"] * 50)
            print(f"  {u['dog']:<35s} over {u['fav']:<25s} {u['prob']:.1%} {bar}")
except Exception:
    traceback.print_exc()

# COMMAND ----------

# Verification
print("\n" + "=" * 60)
print("PREDICTION TABLES")
print("=" * 60)
for t in ["pairwise_probabilities", "bracket_simulation", "model_metadata", "trained_models"]:
    try:
        c = spark.table(f"bracketology.predictions.{t}").count()
        print(f"  bracketology.predictions.{t}: {c:,} rows")
    except Exception as e:
        print(f"  bracketology.predictions.{t}: {e}")
print("=" * 60)
