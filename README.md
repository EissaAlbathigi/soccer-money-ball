# Soccer Moneyball

Soccer Moneyball is a data science project that uses player statistics to find potentially undervalued soccer players in Europe's Big Five leagues.

I combined Transfermarkt market values with FBref performance data from the 2024-25 season. I then built a machine learning model that estimates player values based on their performance.

## Live Dashboard

**https://soccer-money-ball.streamlit.app **

The dashboard lets users filter players by league, position, age, market value, and playing time. Users can also compare players and see how the model's estimated value compares to their actual market value.

## How It Works

The final dataset includes 1,357 players. The model uses information such as:

- Age and minutes played
- Goals and assists
- Expected goals (xG) and expected assists (xAG)
- Progressive passes and carries
- Shots and position
- Per-90 statistics

I trained a Random Forest model to estimate each player's value and calculated:

**Value Gap = Model Value - Market Value**

A positive value gap means the model estimates the player to be worth more than their current market value. I also used 5-fold out-of-fold predictions so players are ranked using predictions from models that were not trained on them.

## Results

- **Mean Absolute Error (MAE):** €9.60M
- **R²:** 0.462

The goal is not to predict an exact price for every player. Instead, the model helps narrow down a large group of players to find interesting recruitment targets.

## Tools 

Python, pandas, scikit-learn, Streamlit, and matplotlib.

## Run 

Install the requirements:

```bash
pip install -r requirements.txt