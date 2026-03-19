# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 4: Model Training & Optimization
# MAGIC
# MAGIC **This is the educational heart of the project.** We walk through each optimization
# MAGIC decision — not just *what* we do, but *why* each step matters.
# MAGIC
# MAGIC ### The Central Tension
# MAGIC
# MAGIC We have ~670 tournament games (2016-2025) to train on, with 10+ features.
# MAGIC This is a **small-data regime** where overfitting is the primary enemy.
# MAGIC Every modeling choice must be understood through this lens.
# MAGIC
# MAGIC ### Roadmap
# MAGIC
# MAGIC | Step | What | Why |
# MAGIC |------|------|-----|
# MAGIC | 1 | Loss function: Log Loss | Measures probability quality, not just accuracy |
# MAGIC | 2 | Baseline: Seed-only logistic regression | Defines the bar to beat |
# MAGIC | 3 | Full-feature logistic regression | Linear model with all features |
# MAGIC | 4 | Cross-validation: LOTO-CV | Honest evaluation without temporal leakage |
# MAGIC | 5 | XGBoost | Non-linear patterns with regularization |
# MAGIC | 6 | Hyperparameter tuning (Optuna) | Bayesian search over model space |
# MAGIC | 7 | Probability calibration | Prevent catastrophic log loss penalties |
# MAGIC | 8 | Ensemble | Combine models to reduce variance |
# MAGIC | 9 | Regularization summary | How we avoided overfitting at every step |

# COMMAND ----------

# MAGIC %pip install scikit-learn xgboost optuna matplotlib seaborn
# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
import xgboost as xgb
import optuna
import json
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

# COMMAND ----------

# Load matchup features
matchup_df = spark.table("bracketology.features.matchup_features").toPandas()
print(f"Training data: {len(matchup_df)} tournament games")
print(f"Seasons: {sorted(matchup_df['season'].unique())}")

# Define feature columns
FEATURE_COLS = [
    "elo_diff", "seed_diff", "win_pct_diff", "avg_margin_diff",
    "sos_diff", "close_game_diff", "away_win_diff",
    "consistency_diff", "momentum_diff",
    "team_a_elo", "team_b_elo", "team_a_seed", "team_b_seed",
]

# Only use features that exist in our data
FEATURE_COLS = [c for c in FEATURE_COLS if c in matchup_df.columns]
print(f"Features: {FEATURE_COLS}")

X = matchup_df[FEATURE_COLS].fillna(0).values
y = matchup_df["team_a_won"].values
seasons = matchup_df["season"].values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Loss Function — Why Log Loss?
# MAGIC
# MAGIC ### The Problem with Accuracy
# MAGIC
# MAGIC Accuracy treats all predictions equally: "I was 51% sure Team A wins" and
# MAGIC "I was 99% sure Team A wins" both count the same if Team A wins.
# MAGIC
# MAGIC **Log loss penalizes confident wrong predictions exponentially.**
# MAGIC
# MAGIC ```
# MAGIC Log Loss = -[y * log(p) + (1-y) * log(1-p)]
# MAGIC ```
# MAGIC
# MAGIC | Prediction | Outcome | Log Loss | Accuracy |
# MAGIC |-----------|---------|----------|----------|
# MAGIC | P(win) = 0.95 | Win | 0.05 | 1 |
# MAGIC | P(win) = 0.55 | Win | 0.60 | 1 |
# MAGIC | P(win) = 0.95 | **Loss** | **3.00** | 0 |
# MAGIC | P(win) = 0.55 | Loss | 0.60 | 0 |
# MAGIC
# MAGIC Being 95% confident and wrong costs **5x more** than being 55% confident and wrong.
# MAGIC This is exactly what we want — it rewards **calibrated** probabilities.
# MAGIC
# MAGIC > **Key insight**: A model that says "1-seed beats 16-seed 96% of the time" is
# MAGIC > better than one that says "100% of the time" — because 1-seeds actually lose ~1.5% of the time.
# MAGIC > The 100%-confident model gets destroyed when the upset happens.

# COMMAND ----------

# Demonstrate log loss behavior
demo_probs = np.linspace(0.01, 0.99, 100)
ll_correct = -np.log(demo_probs)  # Log loss when outcome matches prediction
ll_wrong = -np.log(1 - demo_probs)  # Log loss when outcome doesn't match

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(demo_probs, ll_correct, label="Predicted correctly", color="green", linewidth=2)
ax.plot(demo_probs, ll_wrong, label="Predicted incorrectly", color="red", linewidth=2)
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Log Loss (lower is better)")
ax.set_title("Log Loss: Why Calibration Matters More Than Confidence")
ax.legend(fontsize=12)
ax.axhline(y=0.693, color="gray", linestyle="--", alpha=0.5, label="Random guess (0.5)")
ax.annotate("Confident & wrong\n= catastrophic", xy=(0.95, 3), fontsize=11,
            arrowprops=dict(arrowstyle="->"), xytext=(0.7, 3.5))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Baseline — Seed-Only Logistic Regression
# MAGIC
# MAGIC This is **the chalk bracket in mathematical form**. If you can't beat this model,
# MAGIC all your fancy features are noise.
# MAGIC
# MAGIC A single feature — `seed_diff` — captures the tournament committee's assessment.

# COMMAND ----------

# Leave-One-Tournament-Out Cross Validation (LOTO-CV)
def loto_cv(model_fn, X, y, seasons, feature_cols=None):
    """
    Train on all years except one, predict that year, repeat.
    Returns predictions aligned with the original data.
    """
    unique_seasons = sorted(set(seasons))
    all_preds = np.zeros(len(y))
    all_actuals = np.zeros(len(y))
    results_by_season = {}

    for test_season in unique_seasons:
        train_mask = seasons != test_season
        test_mask = seasons == test_season

        if test_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]

        all_preds[test_mask] = preds
        all_actuals[test_mask] = y_test

        season_ll = log_loss(y_test, preds)
        season_acc = accuracy_score(y_test, (preds > 0.5).astype(int))
        results_by_season[test_season] = {"log_loss": season_ll, "accuracy": season_acc, "games": test_mask.sum()}

    # Overall metrics (only on games that were predicted)
    predicted_mask = all_preds > 0
    overall_ll = log_loss(all_actuals[predicted_mask], all_preds[predicted_mask])
    overall_acc = accuracy_score(all_actuals[predicted_mask], (all_preds[predicted_mask] > 0.5).astype(int))

    return {
        "log_loss": overall_ll,
        "accuracy": overall_acc,
        "predictions": all_preds,
        "by_season": results_by_season
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why LOTO-CV Instead of Random K-Fold?
# MAGIC
# MAGIC Random K-fold would mix games from 2018 and 2023 in the same fold.
# MAGIC This creates **temporal leakage**: the model could learn "2018 patterns" from training
# MAGIC data and exploit them in the same year's test set.
# MAGIC
# MAGIC LOTO-CV simulates the real prediction scenario: train on all past tournaments, predict the next one.

# COMMAND ----------

# Baseline: seed_diff only
seed_idx = FEATURE_COLS.index("seed_diff") if "seed_diff" in FEATURE_COLS else 0
X_seed_only = X[:, seed_idx:seed_idx+1]

baseline_results = loto_cv(
    lambda: LogisticRegression(C=1.0, max_iter=1000),
    X_seed_only, y, seasons
)

print("=" * 50)
print("BASELINE: Seed-Only Logistic Regression")
print("=" * 50)
print(f"  LOTO-CV Log Loss: {baseline_results['log_loss']:.4f}")
print(f"  LOTO-CV Accuracy: {baseline_results['accuracy']:.1%}")
print("\n  Per-season breakdown:")
for season, metrics in sorted(baseline_results["by_season"].items()):
    print(f"    {season}: LL={metrics['log_loss']:.4f}, Acc={metrics['accuracy']:.1%} ({metrics['games']} games)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Full Feature Logistic Regression
# MAGIC
# MAGIC Now let's add all our engineered features. Logistic regression is a good
# MAGIC starting point because:
# MAGIC - It's interpretable (coefficients tell us feature importance)
# MAGIC - It outputs calibrated probabilities naturally
# MAGIC - It's hard to overfit with regularization

# COMMAND ----------

# Scale features for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

full_lr_results = loto_cv(
    lambda: LogisticRegression(C=0.1, max_iter=1000, penalty="l2"),
    X_scaled, y, seasons
)

print("=" * 50)
print("FULL FEATURE: Logistic Regression (L2 regularized)")
print("=" * 50)
print(f"  LOTO-CV Log Loss: {full_lr_results['log_loss']:.4f}")
print(f"  LOTO-CV Accuracy: {full_lr_results['accuracy']:.1%}")
print(f"  Improvement over baseline: {baseline_results['log_loss'] - full_lr_results['log_loss']:.4f} log loss")

# Show coefficients from final model trained on all data
lr_model = LogisticRegression(C=0.1, max_iter=1000, penalty="l2")
lr_model.fit(X_scaled, y)

coef_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": lr_model.coef_[0],
    "abs_coef": np.abs(lr_model.coef_[0])
}).sort_values("abs_coef", ascending=False)

print("\nFeature coefficients (magnitude = importance):")
for _, row in coef_df.iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    print(f"  {direction} {row['feature']:<25s} {row['coefficient']:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Cross-Validation Deep Dive
# MAGIC
# MAGIC Let's visualize how our model performs across different tournament years.
# MAGIC Years with more upsets will naturally have higher log loss.

# COMMAND ----------

# Compare baseline vs full model per season
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

seasons_sorted = sorted(baseline_results["by_season"].keys())
baseline_lls = [baseline_results["by_season"][s]["log_loss"] for s in seasons_sorted]
full_lls = [full_lr_results["by_season"].get(s, {}).get("log_loss", 0) for s in seasons_sorted]

x = range(len(seasons_sorted))
width = 0.35

axes[0].bar([i - width/2 for i in x], baseline_lls, width, label="Seed Only", color="#e74c3c", alpha=0.7)
axes[0].bar([i + width/2 for i in x], full_lls, width, label="Full Features", color="#2ecc71", alpha=0.7)
axes[0].set_xticks(x)
axes[0].set_xticklabels([str(int(s)) for s in seasons_sorted], rotation=45)
axes[0].set_ylabel("Log Loss (lower = better)")
axes[0].set_title("Log Loss by Tournament Year")
axes[0].legend()

baseline_accs = [baseline_results["by_season"][s]["accuracy"] for s in seasons_sorted]
full_accs = [full_lr_results["by_season"].get(s, {}).get("accuracy", 0) for s in seasons_sorted]

axes[1].bar([i - width/2 for i in x], baseline_accs, width, label="Seed Only", color="#e74c3c", alpha=0.7)
axes[1].bar([i + width/2 for i in x], full_accs, width, label="Full Features", color="#2ecc71", alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(int(s)) for s in seasons_sorted], rotation=45)
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy by Tournament Year")
axes[1].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: XGBoost — Gradient Boosted Trees
# MAGIC
# MAGIC ### Why Trees After Logistic Regression?
# MAGIC
# MAGIC Logistic regression assumes **linear** relationships between features and win probability.
# MAGIC But some relationships are non-linear:
# MAGIC - A 1-seed vs 16-seed is fundamentally different from a 5-seed vs 12-seed
# MAGIC - The "value" of an Elo advantage changes depending on absolute Elo level
# MAGIC
# MAGIC **XGBoost** can capture these patterns with **shallow trees** (depth 2-4),
# MAGIC which limits complexity and helps prevent overfitting.
# MAGIC
# MAGIC ### Key Regularization Knobs
# MAGIC
# MAGIC | Parameter | What it does | Our strategy |
# MAGIC |-----------|-------------|--------------|
# MAGIC | `max_depth` | Tree depth | 2-4 (shallow = less overfitting) |
# MAGIC | `learning_rate` | Step size | 0.01-0.1 (slow = more robust) |
# MAGIC | `n_estimators` | Number of trees | 50-500 (with early stopping) |
# MAGIC | `subsample` | Data fraction per tree | 0.6-0.9 (adds randomness) |
# MAGIC | `colsample_bytree` | Feature fraction per tree | 0.6-0.9 |
# MAGIC | `reg_alpha` (L1) | Feature sparsity | Encourages simpler models |
# MAGIC | `reg_lambda` (L2) | Weight decay | Prevents large coefficients |

# COMMAND ----------

# XGBoost with conservative hyperparameters
xgb_results = loto_cv(
    lambda: xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    ),
    X, y, seasons
)

print("=" * 50)
print("XGBOOST: Gradient Boosted Trees")
print("=" * 50)
print(f"  LOTO-CV Log Loss: {xgb_results['log_loss']:.4f}")
print(f"  LOTO-CV Accuracy: {xgb_results['accuracy']:.1%}")
print(f"  vs Baseline: {baseline_results['log_loss'] - xgb_results['log_loss']:+.4f}")
print(f"  vs Full LR:  {full_lr_results['log_loss'] - xgb_results['log_loss']:+.4f}")

# COMMAND ----------

# Feature importance from XGBoost
xgb_model = xgb.XGBClassifier(
    max_depth=3, learning_rate=0.05, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42, use_label_encoder=False,
)
xgb_model.fit(X, y)

importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance["feature"], importance["importance"], color=sns.color_palette("viridis", len(importance)))
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("XGBoost Feature Importance — What Actually Drives Predictions?")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Hyperparameter Tuning with Optuna
# MAGIC
# MAGIC ### Why Bayesian Optimization?
# MAGIC
# MAGIC - **Grid search**: tries every combination — exponential in # of parameters
# MAGIC - **Random search**: better coverage but wasteful
# MAGIC - **Bayesian (Optuna)**: learns from past trials which regions of parameter space are promising
# MAGIC
# MAGIC With only ~670 training examples, we must be extra careful:
# MAGIC - Optimize over LOTO-CV (not training loss!)
# MAGIC - Use relatively few trials (50-100) to avoid "overfitting the validation"

# COMMAND ----------

def objective(trial):
    """Optuna objective: minimize LOTO-CV log loss for XGBoost."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    result = loto_cv(
        lambda: xgb.XGBClassifier(
            **params, eval_metric="logloss", random_state=42, use_label_encoder=False
        ),
        X, y, seasons
    )

    return result["log_loss"]

# Run optimization
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=75, show_progress_bar=True)

print(f"\nBest LOTO-CV Log Loss: {study.best_value:.4f}")
print(f"Best parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# Train best XGBoost model on all data
best_xgb = xgb.XGBClassifier(
    **study.best_params,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
)
best_xgb.fit(X, y)

best_xgb_results = loto_cv(
    lambda: xgb.XGBClassifier(
        **study.best_params, eval_metric="logloss", random_state=42, use_label_encoder=False
    ),
    X, y, seasons
)

print(f"Tuned XGBoost LOTO-CV Log Loss: {best_xgb_results['log_loss']:.4f}")
print(f"Tuned XGBoost LOTO-CV Accuracy: {best_xgb_results['accuracy']:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Probability Calibration
# MAGIC
# MAGIC ### The Problem
# MAGIC
# MAGIC XGBoost outputs "probability-like" scores, but they may not be well-calibrated.
# MAGIC A "calibrated" model means: when it says "70% chance", the event should happen ~70% of the time.
# MAGIC
# MAGIC **Platt scaling** (isotonic regression is an alternative) fits a logistic function to
# MAGIC map raw scores to calibrated probabilities.
# MAGIC
# MAGIC ### Why This Prevents Catastrophic Log Loss
# MAGIC
# MAGIC If a model outputs P=0.99 for a 1-seed and the 1-seed loses:
# MAGIC - Log loss = -log(0.01) = **4.6** (catastrophic)
# MAGIC
# MAGIC After calibration, P=0.95:
# MAGIC - Log loss = -log(0.05) = **3.0** (still bad, but survivable)
# MAGIC
# MAGIC Calibration "clips" extreme probabilities to match historical frequencies.

# COMMAND ----------

# Reliability diagram (before calibration)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, (name, preds) in enumerate([
    ("Logistic Regression", full_lr_results["predictions"]),
    ("XGBoost (tuned)", best_xgb_results["predictions"]),
]):
    # Filter out zero predictions (from LOTO-CV gaps)
    mask = preds > 0
    prob_true, prob_pred = calibration_curve(y[mask], preds[mask], n_bins=10)

    axes[i].plot(prob_pred, prob_true, marker="o", linewidth=2, markersize=8, label=name)
    axes[i].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[i].set_xlabel("Mean Predicted Probability")
    axes[i].set_ylabel("Fraction of Positives")
    axes[i].set_title(f"Reliability Diagram: {name}")
    axes[i].legend()
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, 1)

    brier = brier_score_loss(y[mask], preds[mask])
    axes[i].text(0.05, 0.85, f"Brier: {brier:.4f}", transform=axes[i].transAxes, fontsize=11)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Ensemble
# MAGIC
# MAGIC ### Why Ensembles Work
# MAGIC
# MAGIC Different models make different errors. By averaging their predictions,
# MAGIC we smooth out individual model quirks:
# MAGIC
# MAGIC - **Logistic Regression**: good at linear relationships, well-calibrated
# MAGIC - **XGBoost**: captures non-linearities, may overfit slightly
# MAGIC - **Random Forest**: robust, different tree structure than XGBoost
# MAGIC
# MAGIC We optimize the **weights** to minimize LOTO-CV log loss.

# COMMAND ----------

# Random Forest with conservative settings
rf_results = loto_cv(
    lambda: RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=10,
        max_features="sqrt", random_state=42
    ),
    X, y, seasons
)

print(f"Random Forest LOTO-CV Log Loss: {rf_results['log_loss']:.4f}")
print(f"Random Forest LOTO-CV Accuracy: {rf_results['accuracy']:.1%}")

# COMMAND ----------

# Optimize ensemble weights
from scipy.optimize import minimize

# Get predictions from all three models
pred_lr = full_lr_results["predictions"]
pred_xgb = best_xgb_results["predictions"]
pred_rf = rf_results["predictions"]

# Stack predictions
pred_stack = np.column_stack([pred_lr, pred_xgb, pred_rf])

# Only use games where all models made predictions
valid_mask = (pred_lr > 0) & (pred_xgb > 0) & (pred_rf > 0)

def ensemble_loss(weights):
    """Log loss for weighted ensemble."""
    w = np.array(weights)
    w = w / w.sum()  # Normalize
    blended = pred_stack[valid_mask] @ w
    blended = np.clip(blended, 0.001, 0.999)
    return log_loss(y[valid_mask], blended)

# Optimize weights
result = minimize(
    ensemble_loss,
    x0=[0.33, 0.34, 0.33],  # Start with equal weights
    method="Nelder-Mead",
    bounds=[(0.05, 0.9)] * 3,
)

optimal_weights = result.x / result.x.sum()
print(f"Optimal ensemble weights:")
print(f"  Logistic Regression: {optimal_weights[0]:.3f}")
print(f"  XGBoost:             {optimal_weights[1]:.3f}")
print(f"  Random Forest:       {optimal_weights[2]:.3f}")

# Ensemble predictions
ensemble_preds = pred_stack @ optimal_weights
ensemble_preds_clipped = np.clip(ensemble_preds, 0.001, 0.999)

ensemble_ll = log_loss(y[valid_mask], ensemble_preds_clipped[valid_mask])
ensemble_acc = accuracy_score(y[valid_mask], (ensemble_preds_clipped[valid_mask] > 0.5).astype(int))

print(f"\nEnsemble LOTO-CV Log Loss: {ensemble_ll:.4f}")
print(f"Ensemble LOTO-CV Accuracy: {ensemble_acc:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Results Summary & Model Comparison

# COMMAND ----------

# Final comparison
print("=" * 65)
print("MODEL COMPARISON (LOTO-CV)")
print("=" * 65)
print(f"{'Model':<35s} {'Log Loss':>10s} {'Accuracy':>10s}")
print("-" * 65)
print(f"{'Seed-Only Baseline':<35s} {baseline_results['log_loss']:>10.4f} {baseline_results['accuracy']:>10.1%}")
print(f"{'Full-Feature Logistic Regression':<35s} {full_lr_results['log_loss']:>10.4f} {full_lr_results['accuracy']:>10.1%}")
print(f"{'XGBoost (default params)':<35s} {xgb_results['log_loss']:>10.4f} {xgb_results['accuracy']:>10.1%}")
print(f"{'XGBoost (Optuna-tuned)':<35s} {best_xgb_results['log_loss']:>10.4f} {best_xgb_results['accuracy']:>10.1%}")
print(f"{'Random Forest':<35s} {rf_results['log_loss']:>10.4f} {rf_results['accuracy']:>10.1%}")
print(f"{'Weighted Ensemble':<35s} {ensemble_ll:>10.4f} {ensemble_acc:>10.1%}")
print("=" * 65)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regularization Summary
# MAGIC
# MAGIC Every step included protection against overfitting:
# MAGIC
# MAGIC | Technique | Where Applied | Effect |
# MAGIC |-----------|--------------|--------|
# MAGIC | **L2 penalty** | Logistic Regression (C=0.1) | Shrinks coefficients toward zero |
# MAGIC | **Shallow trees** | XGBoost (depth 2-5) | Limits model complexity |
# MAGIC | **Subsampling** | XGBoost (subsample, colsample) | Adds randomness, reduces variance |
# MAGIC | **L1 + L2 regularization** | XGBoost (alpha, lambda) | Sparsity + weight decay |
# MAGIC | **LOTO-CV** | All models | Prevents temporal leakage |
# MAGIC | **Bayesian tuning (Optuna)** | XGBoost | Efficient search, fewer trials = less overfitting |
# MAGIC | **Calibration** | Ensemble | Prevents extreme probabilities |
# MAGIC | **Ensembling** | Final output | Averages out individual model errors |

# COMMAND ----------

# Save models and metadata
# Train final models on ALL data
final_lr = LogisticRegression(C=0.1, max_iter=1000, penalty="l2")
final_lr.fit(X_scaled, y)

final_xgb = xgb.XGBClassifier(
    **study.best_params, eval_metric="logloss", random_state=42, use_label_encoder=False
)
final_xgb.fit(X, y)

final_rf = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=10, max_features="sqrt", random_state=42
)
final_rf.fit(X, y)

# Store model metadata as JSON in a Delta table
model_metadata = {
    "feature_cols": FEATURE_COLS,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "ensemble_weights": optimal_weights.tolist(),
    "baseline_log_loss": float(baseline_results["log_loss"]),
    "ensemble_log_loss": float(ensemble_ll),
    "ensemble_accuracy": float(ensemble_acc),
    "best_xgb_params": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in study.best_params.items()},
    "lr_coefficients": {feat: float(coef) for feat, coef in zip(FEATURE_COLS, lr_model.coef_[0])},
    "lr_intercept": float(lr_model.intercept_[0]),
    "training_games": int(len(y)),
    "training_seasons": sorted([int(s) for s in set(seasons)]),
}

# Save metadata as key-value Delta table
metadata_rows = [{"key": k, "value": json.dumps(v)} for k, v in model_metadata.items()]
metadata_df = spark.createDataFrame(pd.DataFrame(metadata_rows))
metadata_df.write.mode("overwrite").saveAsTable("bracketology.predictions.model_metadata")

# Save XGBoost model natively (JSON format — no pickle needed)
xgb_model_json = final_xgb.get_booster().save_raw(raw_format="json").decode("utf-8")
xgb_model_rows = [{"model_name": "xgboost", "model_format": "json", "model_data": xgb_model_json}]
xgb_model_df = spark.createDataFrame(pd.DataFrame(xgb_model_rows))
xgb_model_df.write.mode("overwrite").saveAsTable("bracketology.predictions.trained_models")

print("Models and metadata saved to bracketology.predictions.*")
print(f"  Model metadata keys: {list(model_metadata.keys())}")
print(f"  XGBoost model saved as native JSON ({len(xgb_model_json):,} chars)")
print(f"  LR model saved as coefficients + intercept in metadata")
print(f"  Scaler parameters saved in metadata for inference")
