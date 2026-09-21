"""Prepare a first real-data modeling table from Transfermarkt-derived CSVs.

Expected files in data/raw:
  players.csv or players.csv.gz
  player_valuations.csv or player_valuations.csv.gz
  appearances.csv or appearances.csv.gz
  games.csv or games.csv.gz

The maintained public dataset is documented at:
https://github.com/dcaribou/transfermarkt-datasets

This starter deliberately uses fields common to that dataset and avoids scraping Transfermarkt directly.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "transfermarkt_players_modeling.csv"


def locate(stem: str) -> Path:
    for name in [f"{stem}.csv", f"{stem}.csv.gz"]:
        p = RAW / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Put {stem}.csv or {stem}.csv.gz in {RAW}")


def age_from_dob(dob: pd.Series, reference_date: pd.Timestamp) -> pd.Series:
    d = pd.to_datetime(dob, errors="coerce")
    return ((reference_date - d).dt.days / 365.25).round(1)


def main():
    players = pd.read_csv(locate("players"))
    valuations = pd.read_csv(locate("player_valuations"))
    apps = pd.read_csv(locate("appearances"))
    games = pd.read_csv(locate("games"))

    # Normalize common naming variants.
    if "name" in players.columns and "player_name" not in players.columns:
        players = players.rename(columns={"name": "player_name"})
    if "market_value_in_eur" in valuations.columns and "market_value_eur" not in valuations.columns:
        valuations = valuations.rename(columns={"market_value_in_eur": "market_value_eur"})

    # Current/latest valuation for each player.
    date_col = "date" if "date" in valuations.columns else None
    if date_col:
        valuations[date_col] = pd.to_datetime(valuations[date_col], errors="coerce")
        latest_val = valuations.sort_values(date_col).groupby("player_id", as_index=False).tail(1)
    else:
        latest_val = valuations.groupby("player_id", as_index=False).tail(1)

    # Attach dates to appearances to keep the most recent completed ~365 days.
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    latest_date = games["date"].max()
    cutoff = latest_date - pd.Timedelta(days=365)
    recent_game_ids = games.loc[games["date"] >= cutoff, "game_id"]
    apps = apps[apps["game_id"].isin(recent_game_ids)].copy()

    agg_map = {}
    for c in ["minutes_played", "goals", "assists", "yellow_cards", "red_cards"]:
        if c in apps.columns:
            agg_map[c] = "sum"
    if not agg_map:
        raise ValueError("Appearance table does not contain expected performance columns.")
    perf = apps.groupby("player_id", as_index=False).agg(agg_map)
    if "minutes_played" not in perf.columns:
        raise ValueError("minutes_played is required for the starter model.")

    perf["minutes"] = perf["minutes_played"]
    perf["starts"] = np.nan  # optional in real-data starter
    for stat in ["goals", "assists"]:
        if stat not in perf.columns:
            perf[stat] = 0
        perf[f"{stat}_p90"] = perf[stat] / perf["minutes"].clip(lower=1) * 90

    cols = [c for c in ["player_id", "player_name", "position", "date_of_birth", "international_caps"] if c in players.columns]
    df = players[cols].merge(perf, on="player_id", how="inner")
    keep_val = [c for c in ["player_id", "market_value_eur"] if c in latest_val.columns]
    df = df.merge(latest_val[keep_val], on="player_id", how="inner")
    if "date_of_birth" in df.columns:
        df["age"] = age_from_dob(df["date_of_birth"], latest_date)
    else:
        df["age"] = np.nan

    # Starter proxies. Replace with FBref advanced metrics in v2.
    df["progressive_actions_p90"] = df["assists_p90"] * 4 + df["goals_p90"] * 1.5
    df["defensive_actions_p90"] = 0.0
    df["league_strength"] = 1.0
    if "international_caps" not in df.columns:
        df["international_caps"] = 0

    final_cols = [
        "player_id", "player_name", "position", "age", "minutes", "starts",
        "goals_p90", "assists_p90", "progressive_actions_p90",
        "defensive_actions_p90", "league_strength", "international_caps",
        "market_value_eur"
    ]
    df = df[final_cols].dropna(subset=["market_value_eur", "minutes", "position"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} player rows to {OUT}")
    print("Note: this is a starter real-data table. Add FBref advanced metrics for the strongest portfolio version.")


if __name__ == "__main__":
    main()
