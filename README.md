# Soccer Moneyball: Finding Undervalued Players

A data-science + business analytics portfolio project that asks a practical football recruitment question:

> **Which players appear undervalued relative to their observable performance, and how can a club allocate a limited transfer budget more efficiently?**

## Why this project matters

Football clubs make high-stakes capital-allocation decisions in the transfer market. This project treats player recruitment as both a **prediction problem** (estimate fair market value) and a **business decision problem** (identify attractive value opportunities under budget and roster constraints).

The project is designed to demonstrate internship-relevant skills:

- Python and pandas data cleaning
- Feature engineering
- Regression / machine learning
- Model evaluation
- Business-oriented KPI design
- Ranking and recommendation logic
- Data visualization
- Communicating limitations and recommendations
- Git/GitHub project organization

## Core idea

1. Build a player-level dataset containing age, position, playing time and performance.
2. Train a model to estimate market value.
3. Compare predicted value with observed market value.
4. Compute a modeled **value gap** and **undervaluation percentage**.
5. Rank potential recruitment targets.
6. Apply budget, age and positional constraints to recommend targets.

### Main metrics

```text
Value Gap = Predicted Market Value - Observed Market Value

Undervaluation % = Value Gap / Observed Market Value × 100
```

A large positive gap is a **screening signal**, not proof that a player is truly mispriced.

## Repository structure

```text
soccer_moneyball/
├── app.py                         # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                       # Real downloaded source tables
│   └── processed/                 # Modeling datasets
├── src/
│   ├── generate_demo_data.py      # Creates synthetic demo data
│   ├── prepare_transfermarkt.py   # Creates starter real-data table
│   ├── train_model.py             # Trains model + ranks bargains
│   └── recommend.py               # Budget-constrained recommendations
├── models/
└── outputs/
    ├── model_metrics.json
    ├── player_valuations_scored.csv
    ├── top_undervalued_players.csv
    └── figures/
```

## Quick start: runnable demo

The included demo is completely synthetic and exists so the full workflow can be tested immediately.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_demo_data.py
python src/train_model.py
python src/recommend.py --budget-m 50 --positions Attack Midfield Defender --max-age 25
streamlit run app.py
```

**Important:** Any player named `Demo Player ####` is synthetic. Do not present demo rankings as real football findings.

## Real-data path

A strong public backbone is the maintained `dcaribou/transfermarkt-datasets` project. It provides structured tables for players, historical valuations, appearances, games, transfers and more. Download the required CSV files and place them in `data/raw/`:

```text
players.csv(.gz)
player_valuations.csv(.gz)
appearances.csv(.gz)
games.csv(.gz)
```

Then run:

```bash
python src/prepare_transfermarkt.py
python src/train_model.py --input data/processed/transfermarkt_players_modeling.csv
```

### Recommended V2 improvement

The starter Transfermarkt pipeline uses basic appearance production. For a stronger internship version, merge **FBref advanced performance metrics** such as xG, xA, progressive passing/carrying, shot-creating actions and defensive actions. This makes the valuation model substantially more football-specific.

## Modeling approach

The baseline model is a `RandomForestRegressor` trained on `log(1 + market_value)` because player values are highly right-skewed. The pipeline handles numeric and categorical features and reports:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² on the log-value target

For a polished portfolio version, compare at least three models:

1. Linear / Ridge regression as an interpretable baseline
2. Random Forest
3. Gradient boosting (XGBoost, LightGBM or HistGradientBoosting)

Then explain why the final model was selected instead of simply reporting the highest score.

## Business layer

A club should not simply sign the players with the largest model residual. The recommendation layer adds practical constraints such as:

- transfer budget
- maximum age
- required positions
- minimum playing-time sample
- positive modeled value gap

Future versions can add wages, contract expiry, injury history, league-strength adjustments and homegrown / registration constraints.

## Questions this project can answer

- Which players appear most undervalued by the model?
- Which positions have the largest pricing inefficiencies?
- How strongly do age and performance influence estimated market value?
- Can a club assemble recruitment targets under a fixed budget?
- Does the model systematically overvalue or undervalue certain positions or leagues?

## Interview explanation

A concise way to describe the project:

> I built a football recruitment analytics project that predicts player market value from performance and player characteristics, then converts the model residuals into a business-oriented recruitment screen. I added budget and roster constraints so the output is not just a prediction, but a decision-support tool for allocating transfer capital.

## Limitations

This project should explicitly acknowledge several limitations:

- Transfermarkt market values are estimates, not guaranteed transaction prices.
- A model can reproduce biases that are already embedded in the market-value target.
- Transfer fees also depend on contracts, negotiating leverage, club finances and player preferences.
- Performance differs by league, role and tactical context.
- Model residuals are leads for scouting, not automatic buy/sell recommendations.

Showing these limitations makes the project stronger, not weaker, because it demonstrates business judgment and statistical maturity.

## Suggested portfolio roadmap

**V1:** Complete the provided end-to-end pipeline.  
**V2:** Add real Transfermarkt data.  
**V3:** Merge FBref advanced metrics.  
**V4:** Compare multiple models and add feature importance / SHAP.  
**V5:** Add optimization for multi-player roster construction.  
**V6:** Deploy the Streamlit dashboard and publish a one-page executive summary.

## Data-source note

The starter architecture is based on the publicly documented Transfermarkt-derived dataset maintained by `dcaribou/transfermarkt-datasets`, which provides linked tables for players, appearances and historical valuations. FBref can provide richer on-field performance statistics for the advanced version.
