import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv(
    "data/processed/advanced_players_2425.csv"
)
df["xG_per_90"] = df["xG"] / df["minutes"] * 90
df["xAG_per_90"] = df["xAG"] / df["minutes"] * 90
df["PrgC_per_90"] = df["PrgC"] / df["minutes"] * 90
df["PrgP_per_90"] = df["PrgP"] / df["minutes"] * 90
df["PrgR_per_90"] = df["PrgR"] / df["minutes"] * 90
df["Sh_per_90"] = df["Sh"] / df["minutes"] * 90

# Both models will use these exact same players
y = df["market_value_in_eur"] / 1_000_000


baseline_features = [
    "age",
    "position",
    "minutes",
    "goals",
    "assists",
    "goals_per_90",
    "assists_per_90"
]


advanced_features = baseline_features + [
    "xG",
    "xAG",
    "PrgC",
    "PrgP",
    "PrgR",
    "Sh"
]
per90_features = baseline_features + [
    "xG_per_90",
    "xAG_per_90",
    "PrgC_per_90",
    "PrgP_per_90",
    "PrgR_per_90",
    "Sh_per_90"
]

# Create ONE train/test split and give both models
# the exact same training and testing players.
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42
)


def evaluate_model(features):

    numeric_features = [
        feature
        for feature in features
        if feature != "position"
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "position",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                ["position"]
            )
        ],
        remainder="passthrough"
    )

    model = Pipeline(
        [
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

    X_train = df.loc[train_idx, features]
    X_test = df.loc[test_idx, features]

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return mae, r2


baseline_mae, baseline_r2 = evaluate_model(
    baseline_features
)

advanced_mae, advanced_r2 = evaluate_model(
    advanced_features
)
per90_mae, per90_r2 = evaluate_model(
    per90_features
)

print("\nFAIR MODEL COMPARISON")
print("---------------------")

print(
    f"Baseline Model: MAE €{baseline_mae:.2f}M | "
    f"R² {baseline_r2:.3f}"
)

print(
    f"Advanced Model: MAE €{advanced_mae:.2f}M | "
    f"R² {advanced_r2:.3f}"
)

print("\nImprovement")

print(
    f"MAE reduction: "
    f"€{baseline_mae - advanced_mae:.2f}M"
)

print(
    f"R² increase: "
    f"{advanced_r2 - baseline_r2:.3f}"
)
print(
    f"Per-90 Model: MAE €{per90_mae:.2f}M | "
    f"R² {per90_r2:.3f}"
)