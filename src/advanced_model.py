import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the merged advanced dataset
df = pd.read_csv("data/processed/advanced_players_2425.csv")
df["xG_per_90"] = df["xG"] / df["minutes"] * 90
df["xAG_per_90"] = df["xAG"] / df["minutes"] * 90
df["PrgC_per_90"] = df["PrgC"] / df["minutes"] * 90
df["PrgP_per_90"] = df["PrgP"] / df["minutes"] * 90
df["PrgR_per_90"] = df["PrgR"] / df["minutes"] * 90
df["Sh_per_90"] = df["Sh"] / df["minutes"] * 90
print(f"Players loaded: {len(df)}")

# Use the SAME baseline features, plus advanced performance stats
numeric_features = [
    "age",
    "minutes",
    "goals",
    "assists",
    "goals_per_90",
    "assists_per_90",
    "xG_per_90",
    "xAG_per_90",
    "PrgC_per_90",
    "PrgP_per_90",
    "PrgR_per_90",
    "Sh_per_90"
]

categorical_features = [
    "position"
]

features = numeric_features + categorical_features

X = df[features]
y = df["market_value_in_eur"] / 1_000_000

# Same split settings as baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "position",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

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

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nADVANCED MODEL PERFORMANCE")
print("--------------------------")
print(f"MAE: €{mae:.2f}M")
print(f"R²: {r2:.3f}")

# Predict values for every player
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

df["predicted_value_m"] = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    n_jobs=-1
)
df["actual_value_m"] = df["market_value_in_eur"] / 1_000_000
df["value_gap_m"] = (
    df["predicted_value_m"] - df["actual_value_m"]
)

df["undervaluation_pct"] = (
    df["value_gap_m"] / df["actual_value_m"]
) * 100

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
    "xG",
    "xAG",
    "PrgC",
    "PrgP",
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

bargains.to_csv(
    "outputs/advanced_player_valuations.csv",
    index=False
)

print(
    "\nResults saved to "
    "outputs/advanced_player_valuations.csv"
)