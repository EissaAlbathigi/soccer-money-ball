import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# 1. LOAD DATA
# -----------------------------

df = pd.read_csv("data/processed/real_players_2425.csv")

print(f"Players loaded: {len(df)}")


# -----------------------------
# 2. SELECT FEATURES
# -----------------------------

numeric_features = [
    "age",
    "minutes",
    "goals",
    "assists",
    "goals_per_90",
    "assists_per_90"
]

categorical_features = [
    "position"
]

features = numeric_features + categorical_features

X = df[features]

# Predict market value in millions of euros
y = df["market_value_in_eur"] / 1_000_000


# -----------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -----------------------------
# 4. PREPROCESS DATA
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        ("position", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# -----------------------------
# 5. CREATE MODEL
# -----------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                min_samples_leaf=3
            )
        )
    ]
)


# -----------------------------
# 6. TRAIN MODEL
# -----------------------------

model.fit(X_train, y_train)


# -----------------------------
# 7. TEST MODEL
# -----------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"MAE: €{mae:.2f}M")
print(f"R²: {r2:.3f}")


# -----------------------------
# 8. PREDICT ALL PLAYERS
# -----------------------------

df["predicted_value_m"] = model.predict(X)

df["actual_value_m"] = (
    df["market_value_in_eur"] / 1_000_000
)

df["value_gap_m"] = (
    df["predicted_value_m"] -
    df["actual_value_m"]
)

df["undervaluation_pct"] = (
    df["value_gap_m"] /
    df["actual_value_m"]
) * 100


# -----------------------------
# 9. FIND POTENTIAL BARGAINS
# -----------------------------

bargains = df.sort_values(
    "value_gap_m",
    ascending=False
)

columns_to_show = [
    "player_name",
    "age",
    "position",
    "minutes",
    "goals",
    "assists",
    "actual_value_m",
    "predicted_value_m",
    "value_gap_m",
    "undervaluation_pct"
]

print("\nTOP 20 POTENTIALLY UNDERVALUED PLAYERS")
print("--------------------------------------")

print(
    bargains[columns_to_show]
    .head(20)
    .to_string(index=False)
)


# -----------------------------
# 10. SAVE RESULTS
# -----------------------------

bargains.to_csv(
    "outputs/real_player_valuations.csv",
    index=False
)

print(
    "\nResults saved to "
    "outputs/real_player_valuations.csv"
)