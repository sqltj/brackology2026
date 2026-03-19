# Bracketology

NCAA Men's Basketball Tournament prediction system built on Databricks. Uses ensemble ML models (Logistic Regression, XGBoost, Random Forest) trained on 10 years of tournament data (2016-2025) to predict every game in the 2026 bracket.

## 2026 Predictions (Calibrated)

Championship odds from 10,000 Monte Carlo simulations on the actual bracket:

| Seed | Team | Champ Odds |
|------|------|-----------|
| 1 | Duke | ~41% |
| 1 | Auburn | ~14% |
| 1 | Houston | ~12% |
| 1 | Florida | ~10% |
| 2 | Alabama | ~5% |
| 2 | Tennessee | ~4% |
| 2 | Michigan State | ~3% |
| 2 | St. John's | ~2% |

Key findings:
- **Houston's defense** (elite PPG allowed, rebounding margin) boosts their odds significantly over raw model output
- **Mid-major calibration** fixed — blending with historical seed priors prevents teams like High Point (30-3 in Big South) from being rated above Big Ten teams
- **12-over-5 upsets** remain the most likely upset seed matchup (~35.7% historically)

## Architecture

7 Databricks notebooks forming a pipeline:

```
01_setup_and_data_ingestion    → Raw data from ESPN API → bracketology.raw.*
02_exploratory_analysis        → EDA + visualizations (no tables)
03_feature_engineering         → Elo, efficiency, SOS → bracketology.features.*
04_model_training_and_optimization → Ensemble model + LOTO-CV → bracketology.predictions.model_metadata
05_tournament_predictions      → Pairwise probs + Monte Carlo → bracketology.predictions.*
06_round_update                → Post-round re-optimization (run after each round)
07_bracket_picks               → Calibrated bracket picks on actual 2026 bracket
```

Data flow:
```
ESPN API → raw tables → feature tables → trained models → predictions
                                              ↑
                              tournament results (round update loop)
```

## Setup (Databricks Asset Bundles)

### Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) v0.281.0+
- A Databricks workspace (free edition / starter tier works)
- Serverless compute enabled (default on new workspaces)

### Deploy

```bash
# 1. Clone and configure
git clone <this-repo>
cd bracketology

# 2. Set up Databricks CLI auth (if not already configured)
databricks auth login --host https://<your-workspace>.cloud.databricks.com

# 3. Validate the bundle
databricks bundle validate

# 4. Deploy to dev
databricks bundle deploy

# 5. Run the full pipeline (builds everything from scratch)
databricks bundle run full_pipeline
```

### Run Individual Notebooks

```bash
databricks bundle run data_ingestion       # Just NB01
databricks bundle run feature_engineering  # Just NB03
databricks bundle run bracket_picks        # Just NB07
```

### After Each Tournament Round

```bash
# Pulls results, updates Elo, re-weights ensemble, regenerates picks
databricks bundle run round_refresh
```

### Customize Catalog

By default, data goes to the `bracketology` catalog. To change:

```bash
# Deploy with a different catalog
databricks bundle deploy -t dev --var catalog=my_catalog
```

Or edit `databricks.yml`:
```yaml
targets:
  dev:
    variables:
      catalog: "my_catalog"
```

### Free/Starter Edition Notes

This project is designed for Databricks free and starter tiers:

- **No clusters needed** — all notebooks run on serverless compute
- **No external libraries pre-installed** — each notebook does its own `%pip install`
- **Unity Catalog** — you need to create the `bracketology` catalog manually before first run (NB01 does this via `CREATE CATALOG IF NOT EXISTS`)
- **SQL Warehouse** — the starter warehouse (`6238930edfa85aee`) is used for any SQL operations. Update `warehouse_id` in `databricks.yml` with your own.

## Project Structure

```
bracketology/
├── databricks.yml              # DAB main config
├── resources/
│   └── jobs.yml                # Job definitions (individual + pipeline)
├── src/
│   └── notebooks/              # Databricks notebooks (deployed by DAB)
│       ├── 01_setup_and_data_ingestion.py
│       ├── 02_exploratory_analysis.py
│       ├── 03_feature_engineering.py
│       ├── 04_model_training_and_optimization.py
│       ├── 05_tournament_predictions.py
│       ├── 06_round_update.py
│       ├── 07_bracket_picks.py
│       └── test_connectivity.py
├── notebooks/                  # Local development copies
└── README.md
```

## Methodology

### Feature Engineering
- **Elo ratings** — game-by-game updates with margin-of-victory adjustment and season regression
- **Adjusted efficiency** — offensive and defensive points per possession, adjusted for opponent strength
- **Strength of schedule** — average opponent Elo
- **Situational stats** — close game win %, away/neutral win %, last-10-game momentum
- **Four factors** — turnover rate, free throw rate, 3PT rate, offensive rebound %

### Model Training
- **Leave-One-Tournament-Out CV** (LOTO-CV) — train on all years except one, predict that year. Prevents temporal leakage that random K-fold would introduce.
- **3-model ensemble** — Logistic Regression + XGBoost + Random Forest, weights optimized via scipy minimize on LOTO-CV log loss
- **Probability calibration** — Platt scaling to ensure predicted probabilities match observed frequencies

### Calibration (NB07)
Raw ensemble over-indexes on XGBoost (90% weight), which over-values win percentage without enough SOS penalty. Calibration fixes this:

```
P_calibrated = 0.4 × P_model + 0.6 × P_seed_history + defense_adjustment
```

- **Seed prior**: historical win rates (e.g., 1-seeds beat 16-seeds 98.8%)
- **Defense adjustment**: PPG allowed differential, rebound margin, blocks — up to ±30% shift
- Prevents mid-major bias while still allowing the model to identify genuine upset candidates

### Round-by-Round Re-optimization (NB06)
After each tournament round:
1. Pull actual results from ESPN
2. Update Elo ratings with tournament outcomes
3. Re-weight ensemble (inverse log loss — better-performing model gets more weight)
4. Regenerate predictions for surviving teams only

## Data Sources

- **ESPN Hidden API** — scoreboard, teams, team stats, schedules (free, no auth)
- **Historical**: 10 seasons (2016-2025), ~670 tournament games for training
