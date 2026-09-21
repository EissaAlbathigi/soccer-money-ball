import pandas as pd
import unicodedata
import re

print("Loading raw datasets...")

players = pd.read_csv("data/raw/players.csv")
valuations = pd.read_csv("data/raw/player_valuations.csv")
appearances = pd.read_csv("data/raw/appearances.csv")
games = pd.read_csv("data/raw/games.csv")
advanced = pd.read_csv("data/raw/players_data-2024_2025.csv")

print("Players:", players.shape)
print("Valuations:", valuations.shape)
print("Appearances:", appearances.shape)
print("Games:", games.shape)
print("Advanced stats:", advanced.shape)

print("\nFiltering 2024-25 Big Five matches...")

big_five_ids = ["GB1", "ES1", "L1", "IT1", "FR1"]

games_2425 = games[
    (games["season"] == 2024) &
    (games["competition_id"].isin(big_five_ids))
].copy()

appearances_2425 = appearances.merge(
    games_2425[["game_id", "season"]],
    on="game_id",
    how="inner"
)

print("Big Five games:", games_2425.shape)
print("Big Five appearances:", appearances_2425.shape)
print("Unique players:", appearances_2425["player_id"].nunique())
print("\nAggregating player performance...")

player_stats = appearances_2425.groupby(
    ["player_id", "player_name"],
    as_index=False
).agg(
    minutes=("minutes_played", "sum"),
    goals=("goals", "sum"),
    assists=("assists", "sum")
)

player_stats["goals_per_90"] = (
    player_stats["goals"] / player_stats["minutes"] * 90
)

player_stats["assists_per_90"] = (
    player_stats["assists"] / player_stats["minutes"] * 90
)

print("Player stats:", player_stats.shape)
print(player_stats.head())
print("\nAttaching market values...")

valuations["date"] = pd.to_datetime(valuations["date"])

valuations_2425 = valuations[
    valuations["date"] <= "2025-06-30"
].copy()

latest_values = (
    valuations_2425
    .sort_values("date")
    .groupby("player_id")
    .tail(1)
)

model_data = player_stats.merge(
    latest_values[
        ["player_id", "date", "market_value_in_eur"]
    ],
    on="player_id",
    how="left"
)

print("Model data:", model_data.shape)
print(
    "Missing market values:",
    model_data["market_value_in_eur"].isna().sum()
)
print("\nAttaching player information...")

player_info = players[
    [
        "player_id",
        "date_of_birth",
        "position",
        "current_club_name",
        "current_club_domestic_competition_id"
    ]
].copy()

player_info["date_of_birth"] = pd.to_datetime(
    player_info["date_of_birth"],
    errors="coerce"
)

model_data = model_data.merge(
    player_info,
    on="player_id",
    how="left"
)

print("Model data with player info:", model_data.shape)
print("Missing birth dates:", model_data["date_of_birth"].isna().sum())
print("Positions:")
print(model_data["position"].value_counts(dropna=False))
print("\nCalculating age and applying minimum minutes...")

season_end = pd.Timestamp("2025-06-30")

model_data["age"] = (
    (season_end - model_data["date_of_birth"]).dt.days / 365.25
)

model_data = model_data[
    (model_data["minutes"] >= 900) &
    (model_data["market_value_in_eur"].notna()) &
    (model_data["age"].notna())
].copy()

model_data["actual_value_m"] = (
    model_data["market_value_in_eur"] / 1_000_000
)

print("Players after filtering:", model_data.shape)
print("Minimum minutes:", model_data["minutes"].min())
print("Age range:", model_data["age"].min(), "-", model_data["age"].max())
print("Missing market values:", model_data["actual_value_m"].isna().sum())
print("\nPreparing FBref advanced statistics...")

def clean_name(name):
    name = str(name).lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(
        char for char in name
        if not unicodedata.combining(char)
    )
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


advanced["clean_name"] = advanced["Player"].apply(clean_name)

advanced["position_group"] = (
    advanced["Pos"].str.split(",").str[0]
)

advanced["position_group"] = advanced["position_group"].replace({
    "GK": "Goalkeeper",
    "DF": "Defender",
    "MF": "Midfield",
    "FW": "Attack"
})

print("FBref rows:", advanced.shape[0])
print("Unique cleaned names:", advanced["clean_name"].nunique())

print("Position groups:")
print(advanced["position_group"].value_counts(dropna=False))
print("\nAggregating FBref season statistics...")

advanced["Born"] = pd.to_numeric(
    advanced["Born"],
    errors="coerce"
)

advanced_fixed = advanced.groupby(
    ["Player", "Born", "position_group"],
    as_index=False
).agg(
    Age=("Age", "first"),
    Min=("Min", "sum"),
    Gls=("Gls", "sum"),
    Ast=("Ast", "sum"),
    xG=("xG", "sum"),
    xAG=("xAG", "sum"),
    PrgC=("PrgC", "sum"),
    PrgP=("PrgP", "sum"),
    PrgR=("PrgR", "sum"),
    Sh=("Sh", "sum")
)

advanced_fixed["clean_name"] = (
    advanced_fixed["Player"].apply(clean_name)
)

print("Aggregated FBref rows:", advanced_fixed.shape)
print(
    "Unique cleaned names:",
    advanced_fixed["clean_name"].nunique()
)
print("\nMatching Transfermarkt players to FBref...")

model_data["clean_name"] = (
    model_data["player_name"].apply(clean_name)
)

model_data["birth_year"] = (
    model_data["date_of_birth"].dt.year
)

advanced_selected = advanced_fixed[
    [
        "clean_name",
        "Born",
        "position_group",
        "xG",
        "xAG",
        "PrgC",
        "PrgP",
        "PrgR",
        "Sh"
    ]
].copy()

model_advanced = model_data.merge(
    advanced_selected,
    left_on=[
        "clean_name",
        "birth_year",
        "position"
    ],
    right_on=[
        "clean_name",
        "Born",
        "position_group"
    ],
    how="inner"
)

print("Matched dataset:", model_advanced.shape)
print(
    "Unique Transfermarkt players:",
    model_advanced["player_id"].nunique()
)
print(
    "Duplicate player IDs:",
    model_advanced["player_id"].duplicated().sum()
)
print("\nChecking unmatched Transfermarkt players...")

matched_ids = set(model_advanced["player_id"])

unmatched = model_data[
    ~model_data["player_id"].isin(matched_ids)
].copy()

print("Total unmatched:", unmatched.shape[0])

print(
    unmatched[
        [
            "player_name",
            "birth_year",
            "position",
            "minutes",
            "actual_value_m"
        ]
    ]
    .sort_values("actual_value_m", ascending=False)
    .head(20)
    .to_string(index=False)
)


print("\nRecovering safe one-year birth-year matches...")

unmatched_model = model_data[
    ~model_data["player_id"].isin(model_advanced["player_id"])
].copy()

fallback = unmatched_model.merge(
    advanced_selected,
    left_on=["clean_name", "position"],
    right_on=["clean_name", "position_group"],
    how="inner"
)

fallback["birth_year_difference"] = (
    fallback["birth_year"] - fallback["Born"]
).abs()

safe_fallback = fallback[
    fallback["birth_year_difference"] <= 1
].copy()

print("Safe fallback matches:", safe_fallback.shape[0])

print(
    safe_fallback[
        [
            "player_name",
            "birth_year",
            "Born",
            "position",
            "birth_year_difference"
        ]
    ].to_string(index=False)
)

print("\nAdding safe fallback matches...")

safe_fallback = safe_fallback[
    model_advanced.columns
].copy()

model_advanced = pd.concat(
    [model_advanced, safe_fallback],
    ignore_index=True
)

print("Final matched players:", model_advanced.shape)
print(
    "Unique player IDs:",
    model_advanced["player_id"].nunique()
)
print(
    "Duplicate player IDs:",
    model_advanced["player_id"].duplicated().sum()
)
print("\nAttaching current club and league...")

model_advanced["club"] = model_advanced["current_club_name"]

league_map = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "L1": "Bundesliga",
    "IT1": "Serie A",
    "FR1": "Ligue 1"
}

model_advanced["league"] = (
    model_advanced["current_club_domestic_competition_id"]
    .map(league_map)
)

print("Dataset with current clubs:", model_advanced.shape)
print(
    "Missing clubs:",
    model_advanced["club"].isna().sum()
)
print(
    "Players currently outside Big Five:",
    model_advanced["league"].isna().sum()
)

print("\nCreating per-90 features...")

advanced_stats = [
    "xG",
    "xAG",
    "PrgC",
    "PrgP",
    "PrgR",
    "Sh"
]

for stat in advanced_stats:
    model_advanced[f"{stat}_per_90"] = (
        model_advanced[stat]
        / model_advanced["minutes"]
        * 90
    )

print("Final dataset shape:", model_advanced.shape)

print("\nLeague counts:")
print(model_advanced["league"].value_counts())

print(
    "\nMissing clubs:",
    model_advanced["club"].isna().sum()
)

print(
    "Missing leagues:",
    model_advanced["league"].isna().sum()
)
print("\nRunning final validation checks...")

required_columns = [
    "player_id",
    "player_name",
    "age",
    "position",
    "minutes",
    "market_value_in_eur",
    "xG",
    "xAG",
    "PrgC",
    "PrgP",
    "PrgR",
    "Sh"
]

assert model_advanced["player_id"].is_unique, \
    "Duplicate player IDs found."

assert model_advanced[required_columns].notna().all().all(), \
    "Missing values found in required model columns."

assert len(model_advanced) > 1000, \
    "Unexpectedly small final dataset."

print("Validation passed.")
print("\nSaving rebuilt dataset...")

output_path = "data/processed/advanced_players_2425.csv"

model_advanced.to_csv(
    output_path,
    index=False
)

print("Saved:", output_path)
print("Rows:", len(model_advanced))
print("Columns:", len(model_advanced.columns))

print("\nDataset build complete!")