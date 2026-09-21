import pandas as pd

df = pd.read_csv("outputs/advanced_player_valuations.csv")

# --------------------------------
# CLUB RECRUITMENT REQUIREMENTS
# --------------------------------

MAX_MARKET_VALUE = 40
MAX_AGE = 25
MIN_MINUTES = 1500
POSITION = "Midfield"

# --------------------------------
# FILTER ELIGIBLE PLAYERS
# --------------------------------

targets = df[
    (df["actual_value_m"] <= MAX_MARKET_VALUE) &
    (df["age"] <= MAX_AGE) &
    (df["minutes"] >= MIN_MINUTES) &
    (df["position"] == POSITION) &
    (df["value_gap_m"] > 0)
].copy()

# Rank by model-implied value gap
targets = targets.sort_values(
    "value_gap_m",
    ascending=False
)

columns = [
    "player_name",
    "age",
    "position",
    "minutes",
    "actual_value_m",
    "predicted_value_m",
    "value_gap_m",
    "undervaluation_pct"
]

print("\nCLUB RECRUITMENT PROFILE")
print("------------------------")
print(f"Position needed: {POSITION}")
print(f"Maximum age: {MAX_AGE}")
print(f"Maximum market value: €{MAX_MARKET_VALUE}M")
print(f"Minimum minutes: {MIN_MINUTES}")

print("\nTOP RECRUITMENT TARGETS")
print("-----------------------")

print(
    targets[columns]
    .head(10)
    .to_string(index=False)
)

targets.to_csv(
    "outputs/recruitment_targets.csv",
    index=False
)

print(
    "\nResults saved to outputs/recruitment_targets.csv"
)