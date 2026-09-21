import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Soccer Moneyball Recruitment Tool",
    layout="wide"
)

st.title("Soccer Moneyball")

st.subheader(
    "Data-Driven Recruitment for European Football"
)

st.write(
    """
    This tool identifies potential recruitment opportunities across Europe's
    Big Five leagues by comparing a machine-learning model's expected player
    value with observed market value.

    Use the filters to define a club's recruitment strategy, then explore
    players whose model-implied value exceeds their observed market value.
    """
)

st.caption(
    "2024–25 Season | Big Five European Leagues | "
    "Transfermarkt Market Values + FBref Performance Data"
)

st.divider()

df = pd.read_csv("outputs/advanced_player_valuations.csv")

st.sidebar.header("Recruitment Filters")

positions = ["All"] + sorted(df["position"].dropna().unique().tolist())

league = st.sidebar.selectbox(
    "League",
    ["All"] + sorted(df["league"].dropna().unique().tolist())
)
selected_position = st.sidebar.selectbox(
    "Position",
    positions
)

max_age = st.sidebar.slider(
    "Maximum Age",
    min_value=18,
    max_value=40,
    value=25
)

max_value = st.sidebar.slider(
    "Maximum Market Value (€M)",
    min_value=1,
    max_value=150,
    value=40
)

min_minutes = st.sidebar.slider(
    "Minimum Minutes Played",
    min_value=0,
    max_value=3500,
    value=1500,
    step=100
)

ranking_method = st.sidebar.selectbox(
    "Rank Players By",
    [
        "Absolute Value Gap",
        "Undervaluation Percentage"
    ]
)

filtered = df[
    (df["age"] <= max_age) &
    (df["actual_value_m"] <= max_value) &
    (df["minutes"] >= min_minutes) &
    (df["value_gap_m"] > 0)
].copy()

if league != "All":
    filtered = filtered[filtered["league"] == league]

if selected_position != "All":
    filtered = filtered[
        filtered["position"] == selected_position
    ]

if ranking_method == "Absolute Value Gap":
    filtered = filtered.sort_values(
        "value_gap_m",
        ascending=False
    )
else:
    filtered = filtered.sort_values(
        "undervaluation_pct",
        ascending=False
    )

st.subheader("Top Recruitment Targets")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Players Found",
        len(filtered)
    )

with col2:
    if len(filtered) > 0:
        st.metric(
            "Largest Value Gap",
            f"€{filtered['value_gap_m'].max():.1f}M"
        )

with col3:
    if len(filtered) > 0:
        st.metric(
            "Average Value Gap",
            f"€{filtered['value_gap_m'].mean():.1f}M"
        )


display = filtered.head(25)[
    [
        "player_name",
        "club",
        "league",
        "age",
        "position",
        "minutes",
        "actual_value_m",
        "predicted_value_m",
        "value_gap_m",
        "undervaluation_pct",
        "xG",
        "xAG",
        "PrgC",
        "PrgP"
    ]
].head(25).copy()

display.columns = [
    "Player",
    "Club",
    "League",
    "Age",
    "Position",
    "Minutes",
    "Market Value (€M)",
    "Model Value (€M)",
    "Value Gap (€M)",
    "Undervalued (%)",
    "xG",
    "xAG",
    "Progressive Carries",
    "Progressive Passes"
]

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Age": st.column_config.NumberColumn(
            "Age",
            format="%d"
        ),
        "Minutes": st.column_config.NumberColumn(
            "Minutes",
            format="%d"
        ),
        "Market Value (€M)": st.column_config.NumberColumn(
            "Market Value (€M)",
            format="€%.1fM"
        ),
        "Model Value (€M)": st.column_config.NumberColumn(
            "Model Value (€M)",
            format="€%.1fM"
        ),
        "Value Gap (€M)": st.column_config.NumberColumn(
            "Value Gap (€M)",
            format="€%.1fM"
        ),
        "Undervalued (%)": st.column_config.NumberColumn(
            "Undervalued (%)",
            format="%.1f%%"
        ),
        "xG": st.column_config.NumberColumn(
            "xG",
            format="%.1f"
        ),
        "xAG": st.column_config.NumberColumn(
            "xAG",
            format="%.1f"
        )
    }
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance")
    st.metric("Final Model R²", "0.576")
    st.metric("Final Model MAE", "€9.15M")

with col2:
    st.subheader("Baseline Comparison")
    st.metric("Baseline R²", "0.402")
    st.metric("Baseline MAE", "€10.99M")

st.caption(
    "Model-implied values are recruitment screening signals, not estimates "
    "of actual transfer fees or guaranteed player value."
)
st.divider()

st.subheader("Player Comparison")

comparison_pool = filtered.sort_values(
    "value_gap_m",
    ascending=False
).head(25)

player_options = comparison_pool["player_name"].tolist()

if len(player_options) >= 2:

    col1, col2 = st.columns(2)

    with col1:
        player_1 = st.selectbox(
            "Player 1",
            player_options,
            index=0
        )

    with col2:
        player_2 = st.selectbox(
            "Player 2",
            player_options,
            index=1
        )

    p1 = comparison_pool[
        comparison_pool["player_name"] == player_1
    ].iloc[0]

    p2 = comparison_pool[
        comparison_pool["player_name"] == player_2
    ].iloc[0]

    comparison = pd.DataFrame({
        "Metric": [
            "Age",
            "Market Value (€M)",
            "Model Value (€M)",
            "Value Gap (€M)",
            "Undervaluation (%)",
            "Minutes",
            "xG",
            "xAG",
            "Progressive Carries",
            "Progressive Passes"
        ],
        player_1: [
            round(p1["age"], 1),
            round(p1["actual_value_m"], 1),
            round(p1["predicted_value_m"], 1),
            round(p1["value_gap_m"], 1),
            round(p1["undervaluation_pct"], 1),
            int(p1["minutes"]),
            round(p1["xG"], 1),
            round(p1["xAG"], 1),
            int(p1["PrgC"]),
            int(p1["PrgP"])
        ],
        player_2: [
            round(p2["age"], 1),
            round(p2["actual_value_m"], 1),
            round(p2["predicted_value_m"], 1),
            round(p2["value_gap_m"], 1),
            round(p2["undervaluation_pct"], 1),
            int(p2["minutes"]),
            round(p2["xG"], 1),
            round(p2["xAG"], 1),
            int(p2["PrgC"]),
            int(p2["PrgP"])
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info(
        "Not enough players meet the selected filters for comparison."
    )

st.divider()

st.subheader("Market Value vs. Model-Implied Value")

chart_data = filtered.head(10)[
    [
        "player_name",
        "actual_value_m",
        "predicted_value_m"
    ]
].copy()

chart_data = chart_data.set_index("player_name")

chart_data.columns = [
    "Market Value (€M)",
    "Model-Implied Value (€M)"
]

st.bar_chart(chart_data)