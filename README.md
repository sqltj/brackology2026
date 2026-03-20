# Brackology 2026

NCAA Men's Basketball Tournament prediction system built on Databricks. Uses ensemble ML models (Logistic Regression, XGBoost, Random Forest) trained on 10 years of tournament data (2016-2025) to predict every game in the 2026 bracket.

## Game-by-Game Predictions

All predictions from 10,000 Monte Carlo simulations using calibrated probabilities: `P = 0.4 × model + 0.6 × seed_prior + defense_adjustment`. Probabilities sourced directly from `bracketology.predictions.bracket_simulation_calibrated`.

### First Round (Round of 64)

#### EAST

| Matchup | Pick | Win Prob |
|---------|------|---------|
| [1] Duke vs [16] Siena | **Duke** | 99.2% |
| [8] Ohio State vs [9] TCU | **TCU** | 65.8% UPSET |
| [5] St. John's vs [12] Northern Iowa | **St. John's** | 62.4% |
| [4] Kansas vs [13] CA Baptist | **Kansas** | 52.8% |
| [6] Louisville vs [11] South Florida | **Louisville** | 54.9% |
| [3] Michigan State vs [14] North Dakota State | **Michigan State** | 60.9% |
| [7] UCLA vs [10] UCF | **UCLA** | 64.7% |
| [2] UConn vs [15] Furman | **UConn** | 99.1% |

#### WEST

| Matchup | Pick | Win Prob |
|---------|------|---------|
| [1] Arizona vs [16] Long Island | **Arizona** | 98.9% |
| [8] Villanova vs [9] Utah State | **Utah State** | 64.7% UPSET |
| [5] Wisconsin vs [12] High Point | **High Point** | 75.6% UPSET |
| [4] Arkansas vs [13] Hawai'i | **Arkansas** | 51.9% |
| [6] BYU vs [11] Texas | **BYU** | 82.5% |
| [3] Gonzaga vs [14] Kennesaw State | **Gonzaga** | 98.9% |
| [7] Miami vs [10] Missouri | **Miami** | 90.2% |
| [2] Purdue vs [15] Queens | **Purdue** | 89.9% |

#### SOUTH

| Matchup | Pick | Win Prob |
|---------|------|---------|
| [1] Florida vs [16] Prairie View | **Florida** | 99.1% |
| [8] Clemson vs [9] Iowa | **Clemson** | 61.3% |
| [5] Vanderbilt vs [12] McNeese | **McNeese** | 70.5% UPSET |
| [4] Nebraska vs [13] Troy | **Nebraska** | 99.1% |
| [6] North Carolina vs [11] VCU | **VCU** | 59.4% UPSET |
| [3] Illinois vs [14] Penn | **Illinois** | 98.9% |
| [7] Saint Mary's vs [10] Texas A&M | **Saint Mary's** | 90.5% |
| [2] Houston vs [15] Idaho | **Houston** | 99.1% |

#### MIDWEST

| Matchup | Pick | Win Prob |
|---------|------|---------|
| [1] Michigan vs [16] Howard | **Michigan** | 90.5% |
| [8] Georgia vs [9] Saint Louis | **Saint Louis** | 82.9% UPSET |
| [5] Texas Tech vs [12] Akron | **Akron** | 55.7% UPSET |
| [4] Alabama vs [13] Hofstra | **Alabama** | 58.5% |
| [6] Tennessee vs [11] Miami OH | **Tennessee** | 53.5% |
| [3] Virginia vs [14] Wright State | **Virginia** | 99.1% |
| [7] Kentucky vs [10] Santa Clara | **Santa Clara** | 66.5% UPSET |
| [2] Iowa State vs [15] Tennessee State | **Iowa State** | 99.1% |

### Round of 32

#### EAST

| Matchup | Pick | Win Prob |
|---------|------|---------|
| [1] Duke vs [9] TCU | **Duke** | 98.0% |
| [5] St. John's vs [4] Kansas | **St. John's** | 43.5% |
| [3] Michigan State vs [6] Louisville | **Michigan State** | 57.2% |
| [7] UCLA vs [2] UConn | **UConn** | 95.0% |

#### WEST

| Matchup | Pick | Win Prob |
|---------|------|---------|
| [1] Arizona vs [9] Utah State | **Arizona** | 96.2% |
| [12] High Point vs [4] Arkansas | **High Point** | 52.2% UPSET |
| [6] BYU vs [3] Gonzaga | **Gonzaga** | 91.2% |
| [7] Miami vs [2] Purdue | **Purdue** | 74.8% |

#### SOUTH

| Matchup | Pick | Win Prob |
|---------|------|---------|
| [1] Florida vs [8] Clemson | **Florida** | 68.9% |
| [12] McNeese vs [4] Nebraska | **Nebraska** | 59.3% |
| [11] VCU vs [3] Illinois | **Illinois** | 63.2% |
| [7] Saint Mary's vs [2] Houston | **Houston** | 79.8% |

#### MIDWEST

| Matchup | Pick | Win Prob |
|---------|------|---------|
| [1] Michigan vs [9] Saint Louis | **Michigan** | 81.2% |
| [12] Akron vs [4] Alabama | **Akron** | 34.9% |
| [6] Tennessee vs [3] Virginia | **Virginia** | 70.4% |
| [10] Santa Clara vs [2] Iowa State | **Iowa State** | 95.9% |

### Sweet 16

| Matchup | Pick | Win Prob |
|---------|------|---------|
| [1] Duke vs [2] UConn (EAST) | **Duke** | 89.4% |
| [1] Arizona vs [3] Gonzaga (WEST) | **Arizona** | 92.8% |
| [1] Florida vs [2] Houston (SOUTH) | **Houston** | 71.7% |
| [1] Michigan vs [2] Iowa State (MIDWEST) | **Michigan** | 74.5% |

### Elite 8 (Regional Finals)

| Matchup | Pick | Final Four Prob |
|---------|------|----------------|
| [1] Duke vs [2] UConn (EAST) | **Duke** | 75.0% |
| [1] Arizona vs [3] Gonzaga (WEST) | **Arizona** | 62.8% |
| [2] Houston vs [1] Florida (SOUTH) | **Houston** | 55.6% |
| [1] Michigan vs [2] Iowa State (MIDWEST) | **Michigan** | 47.0% |

### Final Four

| Matchup | Pick | Championship Prob |
|---------|------|------------------|
| [1] Duke (EAST) vs [1] Arizona (WEST) | **Duke** | 51.6% |
| [2] Houston (SOUTH) vs [1] Michigan (MIDWEST) | **Houston** | 33.0% |

### Championship

| Matchup | Pick | Champion Prob |
|---------|------|--------------|
| [1] Duke vs [2] Houston | **Duke** | 41.1% |

### Championship Odds (10,000 Monte Carlo Simulations)

| Seed | Team | Region | R32 | S16 | E8 | F4 | Final | Champ |
|------|------|--------|-----|-----|----|----|-------|-------|
| 1 | Duke | EAST | 99.2% | 98.0% | 89.4% | 75.0% | 51.6% | 41.1% |
| 1 | Arizona | WEST | 98.9% | 96.2% | 92.8% | 62.8% | 28.5% | 18.2% |
| 2 | Houston | SOUTH | 99.1% | 79.8% | 71.7% | 55.6% | 33.0% | 12.2% |
| 1 | Michigan | MIDWEST | 90.5% | 81.2% | 74.5% | 47.0% | 29.9% | 9.4% |
| 2 | UConn | EAST | 99.1% | 95.0% | 86.2% | 21.0% | 9.6% | 6.0% |
| 3 | Gonzaga | WEST | 98.9% | 91.2% | 74.3% | 31.3% | 8.5% | 4.8% |
| 2 | Iowa State | MIDWEST | 99.1% | 95.9% | 55.6% | 30.2% | 11.0% | 3.1% |
| 3 | Virginia | MIDWEST | 99.1% | 70.4% | 36.2% | 15.0% | 7.8% | 1.4% |
| 4 | Nebraska | SOUTH | 99.1% | 59.3% | 35.8% | 10.2% | 4.3% | 1.0% |
| 1 | Florida | SOUTH | 99.1% | 68.9% | 36.1% | 15.7% | 5.6% | 0.9% |
| 7 | Saint Mary's | SOUTH | 90.5% | 20.0% | 15.3% | 8.3% | 3.8% | 0.8% |
| 2 | Purdue | WEST | 89.9% | 74.8% | 21.3% | 4.8% | 0.8% | 0.3% |
| 11 | Miami OH | MIDWEST | 46.5% | 17.3% | 6.4% | 2.4% | 0.9% | 0.2% |
| 12 | McNeese | SOUTH | 70.5% | 34.5% | 17.8% | 5.0% | 1.5% | 0.2% |
| 3 | Michigan State | EAST | 60.9% | 57.2% | 10.4% | 1.9% | 0.5% | 0.2% |
| 3 | Illinois | SOUTH | 98.9% | 63.2% | 11.0% | 3.6% | 0.8% | 0.1% |
| 12 | Northern Iowa | EAST | 37.6% | 17.8% | 3.4% | 0.9% | 0.2% | 0.1% |

### Notable Predictions

- **[12] High Point** (WEST): 75.6% to win R64 over Wisconsin, 52.2% to reach S16 — model's biggest upset call
- **[12] McNeese** (SOUTH): 70.5% to upset Vanderbilt, 34.5% to reach S16
- **[9] Saint Louis** (MIDWEST): 82.9% over Georgia — model treats this as a major mismatch
- **[10] Santa Clara** (MIDWEST): 66.5% over Kentucky
- **[11] VCU** (SOUTH): 59.4% over North Carolina
- **[12] Akron** (MIDWEST): 55.7% over Texas Tech
- **[5] Wisconsin** (WEST): only 24.5% to survive R64 — model's most confident 5-seed upset
- **[5] Vanderbilt** (SOUTH): only 29.5% to survive R64
- **[5] Texas Tech** (MIDWEST): only 44.3% to survive R64
- **[7] Saint Mary's** (SOUTH): 90.5% R64, but 8.3% Final Four — a dark horse Cinderella path
- **[2] Houston** (SOUTH): elite defense pushes them to 12.2% champion despite 2-seed — highest non-1-seed odds

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
git clone https://github.com/sqltj/brackology2026.git
cd brackology2026

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
brackology2026/
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
