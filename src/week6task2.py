import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 1. LOAD DATA


DATA_PATH = "panel_with_liquidity.csv"

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values(["Ticker", "Date"])
      .reset_index(drop=True)
)

print("Data shape:", df.shape)
print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)
print(
    "Number of stocks:",
    df["Ticker"].nunique()
)
print(
    "Number of sectors:",
    df["Sector"].nunique()
)

print("\nSector counts:")
print(df["Sector"].value_counts())



# 2. REQUIRED COLUMNS


required_columns = [
    "Date",
    "Ticker",
    "Return",
    "PE",
    "Sector",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )



# 3. VALUE FACTOR


df["value_score"] = np.nan

valid_pe = (
    df["PE"].notna()
    & (df["PE"] > 0)
)

df.loc[valid_pe, "value_score"] = (
    1.0 / df.loc[valid_pe, "PE"]
)



# 4. 12-1 MONTH MOMENTUM


def momentum_for_stock(returns):

    return (
        (1.0 + returns)
        .rolling(
            window=12,
            min_periods=12
        )
        .apply(
            np.prod,
            raw=True
        )
        .shift(1)
        - 1.0
    )


df["momentum_score"] = (
    df.groupby("Ticker")["Return"]
      .transform(momentum_for_stock)
)



# 5. CROSS-SECTIONAL Z-SCORE


def cross_sectional_zscore(series):

    result = pd.Series(
        np.nan,
        index=series.index
    )

    valid = series.notna()

    if valid.sum() == 0:
        return result

    std = series.loc[valid].std()

    if pd.isna(std) or std == 0:

        result.loc[valid] = 0.0

        return result

    result.loc[valid] = (
        (
            series.loc[valid]
            - series.loc[valid].mean()
        )
        / std
    )

    return result


df["value_z"] = (
    df.groupby("Date")["value_score"]
      .transform(cross_sectional_zscore)
)

df["momentum_z"] = (
    df.groupby("Date")["momentum_score"]
      .transform(cross_sectional_zscore)
)



# 6. 50/50 COMPOSITE


valid = (
    df["value_z"].notna()
    & df["momentum_z"].notna()
)

df["composite_50_50"] = np.nan

df.loc[valid, "composite_50_50"] = (
    0.5 * df.loc[valid, "value_z"]
    + 0.5 * df.loc[valid, "momentum_z"]
)



# 7. RAW TOP QUINTILE


def select_top_quintile(
    df,
    score_column,
    top_fraction=0.20
):

    def select_date_group(group):

        valid_scores = (
            group[score_column].notna()
        )

        selected = pd.Series(
            False,
            index=group.index
        )

        n_valid = valid_scores.sum()

        if n_valid == 0:
            return selected

        n_selected = max(
            1,
            int(
                np.ceil(
                    n_valid
                    * top_fraction
                )
            )
        )

        ranked = (
            group.loc[
                valid_scores,
                score_column
            ]
            .sort_values(
                ascending=False
            )
        )

        selected.loc[
            ranked.index[:n_selected]
        ] = True

        return selected

    return (
        df.groupby(
            "Date",
            group_keys=False
        )
        .apply(
            select_date_group,
            include_groups=False
        )
        .reindex(df.index)
        .fillna(False)
        .astype(bool)
    )


df["raw_selected"] = (
    select_top_quintile(
        df,
        "composite_50_50"
    )
)



# 8. RAW PORTFOLIO WEIGHTS


df["raw_weight"] = 0.0

raw_count = (
    df.groupby("Date")["raw_selected"]
      .transform("sum")
)

valid_raw = (
    df["raw_selected"]
    & (raw_count > 0)
)

df.loc[valid_raw, "raw_weight"] = (
    1.0 / raw_count.loc[valid_raw]
)



# 9. RAW SECTOR EXPOSURE


raw_sector_exposure = (
    df[df["raw_selected"]]
    .groupby(
        ["Date", "Sector"]
    )["raw_weight"]
    .sum()
    .unstack(
        fill_value=0.0
    )
)

print("\nRaw sector exposure:")
print(
    raw_sector_exposure.head()
)



# 10. SECTOR-NEUTRAL RANKING


def select_sector_neutral(
    df,
    score_column,
    top_fraction=0.20
):

    def select_sector_group(group):

        valid_scores = (
            group[score_column].notna()
        )

        selected = pd.Series(
            False,
            index=group.index
        )

        n_valid = valid_scores.sum()

        if n_valid == 0:
            return selected

        n_selected = max(
            1,
            int(
                np.ceil(
                    n_valid
                    * top_fraction
                )
            )
        )

        ranked = (
            group.loc[
                valid_scores,
                score_column
            ]
            .sort_values(
                ascending=False
            )
        )

        selected.loc[
            ranked.index[:n_selected]
        ] = True

        return selected

    return (
        df.groupby(
            ["Date", "Sector"],
            group_keys=False
        )
        .apply(
            select_sector_group,
            include_groups=False
        )
        .reindex(df.index)
        .fillna(False)
        .astype(bool)
    )


df["sector_neutral_selected"] = (
    select_sector_neutral(
        df,
        "composite_50_50"
    )
)



# 11. SECTOR-NEUTRAL WEIGHTS


df["sector_neutral_weight"] = 0.0

sector_counts = (
    df.groupby(
        ["Date", "Sector"]
    )["sector_neutral_selected"]
    .transform("sum")
)

valid_sector = (
    df["sector_neutral_selected"]
    & (sector_counts > 0)
)

# Equal weight within each sector
df.loc[valid_sector, "sector_neutral_weight"] = (
    1.0 / sector_counts.loc[valid_sector]
)



# 12. EQUAL SECTOR WEIGHTING


# The previous step equal-weights stocks within each sector,
# but does not automatically give each sector equal portfolio weight.
#
# To create a genuinely sector-neutral portfolio:
# each active sector receives equal portfolio weight.

sector_selected = (
    df[
        df["sector_neutral_selected"]
    ]
    .groupby(
        ["Date", "Sector"]
    )
    .size()
    .rename("sector_n")
    .reset_index()
)

sector_count_by_date = (
    sector_selected
    .groupby("Date")["Sector"]
    .transform("count")
)

sector_selected["sector_weight"] = (
    1.0 / sector_count_by_date
)

sector_selected["stock_weight"] = (
    sector_selected["sector_weight"]
    / sector_selected["sector_n"]
)


# Merge sector-neutral weights back
df = df.merge(
    sector_selected[
        [
            "Date",
            "Sector",
            "sector_n",
            "sector_weight"
        ]
    ],
    on=["Date", "Sector"],
    how="left"
)

df["sector_neutral_weight"] = np.where(
    df["sector_neutral_selected"],
    df["sector_weight"]
    / df["sector_n"],
    0.0
)

df[
    "sector_neutral_weight"
] = df[
    "sector_neutral_weight"
].fillna(0.0)



# 13. CHECK SECTOR EXPOSURE


sector_neutral_exposure = (
    df[
        df["sector_neutral_selected"]
    ]
    .groupby(
        ["Date", "Sector"]
    )["sector_neutral_weight"]
    .sum()
    .unstack(
        fill_value=0.0
    )
)

print("\nSector-neutral exposure:")
print(
    sector_neutral_exposure.head()
)



# 14. RETURNS MATRIX


returns = (
    df.pivot(
        index="Date",
        columns="Ticker",
        values="Return"
    )
    .sort_index()
)



# 15. WEIGHT MATRICES


raw_weights = (
    df.pivot(
        index="Date",
        columns="Ticker",
        values="raw_weight"
    )
    .fillna(0.0)
    .sort_index()
)

sector_neutral_weights = (
    df.pivot(
        index="Date",
        columns="Ticker",
        values="sector_neutral_weight"
    )
    .fillna(0.0)
    .sort_index()
)



# 16. PORTFOLIO RETURNS


def calculate_portfolio_returns(
    weights,
    returns
):

    lagged_weights = (
        weights
        .reindex(returns.index)
        .fillna(0.0)
        .shift(1)
    )

    lagged_weights = (
        lagged_weights
        .reindex(
            columns=returns.columns,
            fill_value=0.0
        )
    )

    portfolio_returns = (
        returns
        .mul(lagged_weights)
        .sum(axis=1)
    )

    return portfolio_returns


raw_returns = (
    calculate_portfolio_returns(
        raw_weights,
        returns
    )
)

sector_neutral_returns = (
    calculate_portfolio_returns(
        sector_neutral_weights,
        returns
    )
)



# 17. PERFORMANCE STATISTICS


def performance_stats(returns):

    returns = returns.dropna()

    if len(returns) == 0:

        return {
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
            "Observations": 0,
        }

    annualized_return = (
        (1 + returns).prod()
        ** (12 / len(returns))
        - 1
    )

    annualized_volatility = (
        returns.std()
        * np.sqrt(12)
    )

    sharpe = (
        annualized_return
        / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    cumulative = (
        1 + returns
    ).cumprod()

    running_max = (
        cumulative.cummax()
    )

    drawdown = (
        cumulative
        / running_max
        - 1
    )

    return {
        "Annualized Return":
            annualized_return,

        "Annualized Volatility":
            annualized_volatility,

        "Sharpe":
            sharpe,

        "Max Drawdown":
            drawdown.min(),

        "Observations":
            len(returns),
    }


performance_comparison = pd.DataFrame(
    {
        "Raw Composite":
            performance_stats(
                raw_returns
            ),

        "Sector-Neutral Composite":
            performance_stats(
                sector_neutral_returns
            ),
    }
).T


print(
    "\nRaw vs Sector-Neutral Performance:"
)

print(
    performance_comparison.round(4)
)



# 18. CUMULATIVE PERFORMANCE


comparison_returns = pd.DataFrame(
    {
        "Raw Composite":
            raw_returns,

        "Sector-Neutral Composite":
            sector_neutral_returns,
    }
)

cumulative = (
    1 + comparison_returns
).cumprod()


plt.figure(figsize=(12, 6))

for strategy in cumulative.columns:

    plt.plot(
        cumulative.index,
        cumulative[strategy],
        label=strategy
    )

plt.title(
    "Raw vs Sector-Neutral Composite"
)

plt.xlabel("Date")
plt.ylabel("Cumulative Wealth")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()



# 19. SECTOR EXPOSURE CHART


average_raw_exposure = (
    raw_sector_exposure
    .mean()
    .sort_values(
        ascending=False
    )
)

average_neutral_exposure = (
    sector_neutral_exposure
    .mean()
    .reindex(
        average_raw_exposure.index
    )
)

exposure_comparison = pd.DataFrame(
    {
        "Raw": average_raw_exposure,
        "Sector-Neutral":
            average_neutral_exposure,
    }
)

print(
    "\nAverage sector exposure:"
)

print(
    exposure_comparison.round(4)
)


exposure_comparison.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Average Sector Exposure: Raw vs Sector-Neutral"
)

plt.xlabel("Sector")

plt.ylabel("Portfolio Weight")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.show()



# 20. SECTOR EXPOSURE CONCENTRATION


raw_hhi = (
    raw_sector_exposure
    .pow(2)
    .sum(axis=1)
)

neutral_hhi = (
    sector_neutral_exposure
    .pow(2)
    .sum(axis=1)
)

exposure_concentration = pd.DataFrame(
    {
        "Raw HHI": raw_hhi,
        "Sector-Neutral HHI": neutral_hhi,
    }
)

print(
    "\nSector concentration (HHI):"
)

print(
    exposure_concentration.describe()
)



# 21. SAVE OUTPUTS


df[
    [
        "Date",
        "Ticker",
        "Sector",
        "composite_50_50",
        "raw_selected",
        "raw_weight",
        "sector_neutral_selected",
        "sector_neutral_weight",
    ]
].to_csv(
    "task2_sector_neutral_signal_table.csv",
    index=False
)

raw_sector_exposure.to_csv(
    "task2_raw_sector_exposure.csv"
)

sector_neutral_exposure.to_csv(
    "task2_sector_neutral_exposure.csv"
)

performance_comparison.to_csv(
    "task2_performance_comparison.csv"
)

exposure_comparison.to_csv(
    "task2_exposure_comparison.csv"
)

exposure_concentration.to_csv(
    "task2_sector_hhi.csv"
)

comparison_returns.to_csv(
    "task2_portfolio_returns.csv"
)

print(
    "\nTask 2 outputs saved."
)