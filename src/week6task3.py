import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 0. PATH


PROJECT_ROOT = Path.cwd()

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


from attribution import run_factor_regression
from attribution import regression_summary
from attribution import run_attribution



# 1. LOAD DATA


DATA_PATH = "panel_with_liquidity.csv"

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(
    df["Date"]
)

# DATA QUALITY FILTER


df["Return_clean"] = df["Return"].copy()

invalid_return = (
    ~np.isfinite(df["Return"])
    | (df["Return"] <= -1)
    | (df["Return"] > 5)
)

print("\nInvalid return observations:")

print(
    df.loc[
        invalid_return,
        ["Date", "Ticker", "Return", "MarketCap", "Sector"]
    ]
)

print(
    "\nNumber of invalid returns:",
    invalid_return.sum()
)

df.loc[
    invalid_return,
    "Return_clean"
] = np.nan

df = (
    df.sort_values(
        ["Ticker", "Date"]
    )
    .reset_index(drop=True)
)

print(
    "Data shape:",
    df.shape
)

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



# 2. REQUIRED COLUMNS


required_columns = [
    "Date",
    "Ticker",
    "Return",
    "PE",
    "MarketCap",
    "Sector",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )



# 3. VALUE SCORE


df["value_score"] = np.nan

valid_pe = (
    df["PE"].notna()
    & (df["PE"] > 0)
)

df.loc[
    valid_pe,
    "value_score"
] = (
    1.0
    / df.loc[
        valid_pe,
        "PE"
    ]
)



# 4. 12-1 MOMENTUM


def momentum_for_stock(
    returns
):

    return (
        (1.0 + returns)
        .rolling(
            window=12,
            min_periods=12,
        )
        .apply(
            np.prod,
            raw=True,
        )
        .shift(1)
        - 1.0
    )


df["momentum_score"] = (
    df.groupby("Ticker")[
        "Return_clean"
    ]
    .transform(
        momentum_for_stock
    )
)



# 5. CROSS-SECTIONAL Z-SCORE


def cross_sectional_zscore(
    series
):

    result = pd.Series(
        np.nan,
        index=series.index,
    )

    valid = series.notna()

    if valid.sum() == 0:
        return result

    std = (
        series.loc[valid]
        .std()
    )

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
    df.groupby("Date")[
        "value_score"
    ]
    .transform(
        cross_sectional_zscore
    )
)

df["momentum_z"] = (
    df.groupby("Date")[
        "momentum_score"
    ]
    .transform(
        cross_sectional_zscore
    )
)



# 6. 50/50 COMPOSITE


valid_composite = (
    df["value_z"].notna()
    & df["momentum_z"].notna()
)

df["composite_50_50"] = np.nan

df.loc[
    valid_composite,
    "composite_50_50"
] = (
    0.5
    * df.loc[
        valid_composite,
        "value_z"
    ]
    +
    0.5
    * df.loc[
        valid_composite,
        "momentum_z"
    ]
)



# 7. GENERIC TOP / BOTTOM QUINTILE FUNCTION


def quintile_portfolio_returns(
    df,
    score_column,
    return_column="Return_clean",
    q=0.20,
):
    """
    Construct Q1, Q5 and Q5-Q1 factor returns.

    Signal at month t is used to
    form portfolio for month t+1.
    """

    temp = df.copy()

    temp["rank"] = (
        temp.groupby("Date")[
            score_column
        ]
        .rank(
            pct=True,
            method="first",
        )
    )

    temp["top"] = (
        temp["rank"] >= (1 - q)
    )

    temp["bottom"] = (
        temp["rank"] <= q
    )

    # Shift portfolio membership
    temp["top_lag"] = (
        temp.groupby("Ticker")[
            "top"
        ]
        .shift(1)
        .fillna(False)
    )

    temp["bottom_lag"] = (
        temp.groupby("Ticker")[
            "bottom"
        ]
        .shift(1)
        .fillna(False)
    )

    # Top quintile return
    top_returns = (
        temp[
            temp["top_lag"]
        ]
        .groupby("Date")[
            return_column
        ]
        .mean()
    )

    # Bottom quintile return
    bottom_returns = (
        temp[
            temp["bottom_lag"]
        ]
        .groupby("Date")[
            return_column
        ]
        .mean()
    )

    factor_returns = pd.DataFrame(
        {
            "Q5": top_returns,
            "Q1": bottom_returns,
        }
    )

    factor_returns[
        "Q5_Q1"
    ] = (
        factor_returns["Q5"]
        - factor_returns["Q1"]
    )

    return factor_returns



# 8. MARKET FACTOR


# Equal-weighted market return.
#
# This is a simple market proxy and avoids
# using current-period information in the
# portfolio formation process.

market_returns = (
    df.groupby("Date")[
        "Return_clean"
    ]
    .mean()
    .rename("MKT")
)



# 9. SIZE FACTOR (SMB)


# Use previous month's market cap
# to classify stocks into small and large.

df["MarketCap_lag"] = (
    df.groupby("Ticker")[
        "MarketCap"
    ]
    .shift(1)
)

df["size_rank"] = (
    df.groupby("Date")[
        "MarketCap_lag"
    ]
    .rank(
        pct=True,
        method="first",
    )
)

df["Small"] = (
    df["size_rank"] <= 0.50
)

df["Big"] = (
    df["size_rank"] > 0.50
)

small_returns = (
    df[df["Small"]]
    .groupby("Date")[
        "Return_clean"
    ]
    .mean()
)

big_returns = (
    df[df["Big"]]
    .groupby("Date")[
        "Return_clean"
    ]
    .mean()
)

smb = (
    small_returns
    - big_returns
)

smb.name = "SMB"



# 10. VALUE FACTOR


value_factor = (
    quintile_portfolio_returns(
        df,
        "value_score",
    )["Q5_Q1"]
)

value_factor.name = "HML"



# 11. MOMENTUM FACTOR


momentum_factor = (
    quintile_portfolio_returns(
        df,
        "momentum_score",
    )["Q5_Q1"]
)

momentum_factor.name = "MOM"



# 12. FACTOR RETURN TABLE


factor_returns = pd.concat(
    [
        market_returns,
        smb,
        value_factor,
        momentum_factor,
    ],
    axis=1,
)

factor_returns = (
    factor_returns
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .dropna()
)

print(
    "\nFactor returns:"
)

print(
    factor_returns.head()
)

print(
    "\nFactor correlation:"
)

print(
    factor_returns.corr()
    .round(4)
)



# 13. STRATEGY RETURNS



# Raw Composite


def make_top_quintile_weights(
    df,
    score_column,
):

    temp = df.copy()

    temp["rank"] = (
        temp.groupby("Date")[
            score_column
        ]
        .rank(
            pct=True,
            method="first",
        )
    )

    temp["selected"] = (
        temp["rank"] >= 0.80
    )

    temp["weight"] = 0.0

    counts = (
        temp.groupby("Date")[
            "selected"
        ]
        .transform("sum")
    )

    valid = (
        temp["selected"]
        & (counts > 0)
    )

    temp.loc[
        valid,
        "weight"
    ] = (
        1.0
        / counts.loc[valid]
    )

    return (
        temp.pivot(
            index="Date",
            columns="Ticker",
            values="weight",
        )
        .fillna(0.0)
        .sort_index()
    )


raw_weights = (
    make_top_quintile_weights(
        df,
        "composite_50_50",
    )
)



returns_matrix = (
    df.pivot(
        index="Date",
        columns="Ticker",
        values="Return_clean",
    )
    .sort_index()
)

raw_strategy_returns = (
    returns_matrix
    .mul(
        raw_weights
        .reindex(
            returns_matrix.index
        )
        .fillna(0.0)
        .shift(1)
        .reindex(
            columns=returns_matrix.columns,
            fill_value=0.0,
        )
    )
    .sum(axis=1)
)

raw_strategy_returns.name = (
    "Raw_Composite"
)



# 14. SECTOR-NEUTRAL STRATEGY


temp = df.copy()

temp["sector_rank"] = (
    temp.groupby(
        ["Date", "Sector"]
    )["composite_50_50"]
    .rank(
        pct=True,
        method="first",
    )
)

temp["sector_selected"] = (
    temp["sector_rank"] >= 0.80
)

# Number of active sectors
active_sector_count = (
    temp[
        temp["sector_selected"]
    ]
    .groupby("Date")[
        "Sector"
    ]
    .nunique()
)

temp["sector_stock_count"] = (
    temp.groupby(
        ["Date", "Sector"]
    )["sector_selected"]
    .transform("sum")
)

temp["active_sectors"] = (
    temp["Date"]
    .map(active_sector_count)
)

temp["sector_neutral_weight"] = 0.0

valid = (
    temp["sector_selected"]
    & (
        temp["sector_stock_count"]
        > 0
    )
    & (
        temp["active_sectors"]
        > 0
    )
)

temp.loc[
    valid,
    "sector_neutral_weight"
] = (
    1.0
    / temp.loc[
        valid,
        "active_sectors"
    ]
    / temp.loc[
        valid,
        "sector_stock_count"
    ]
)

sector_neutral_weights = (
    temp.pivot(
        index="Date",
        columns="Ticker",
        values="sector_neutral_weight",
    )
    .fillna(0.0)
    .sort_index()
)

sector_neutral_strategy_returns = (
    returns_matrix
    .mul(
        sector_neutral_weights
        .reindex(
            returns_matrix.index
        )
        .fillna(0.0)
        .shift(1)
        .reindex(
            columns=returns_matrix.columns,
            fill_value=0.0,
        )
    )
    .sum(axis=1)
)

sector_neutral_strategy_returns.name = (
    "Sector_Neutral_Composite"
)



# 15. STRATEGY RETURN TABLE


strategy_returns = pd.concat(
    [
        raw_strategy_returns,
        sector_neutral_strategy_returns,
    ],
    axis=1,
)

print(
    "\nStrategy returns:"
)

print(
    strategy_returns.head()
)



# 16. FOUR-FACTOR ATTRIBUTION


attribution_summary = run_attribution(
    strategy_returns,
    factor_returns,
    hac_lags=3,
)

print(
    "\nFour-factor attribution:"
)

print(
    attribution_summary.round(4)
)



# 17. INDIVIDUAL REGRESSION OUTPUT


for strategy in strategy_returns.columns:

    results = run_factor_regression(
        strategy_returns[strategy],
        factor_returns,
        hac_lags=3,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"{strategy} — OLS Regression"
    )

    print(
        "=" * 70
    )

    print(
        results.summary()
    )



# 18. RESIDUAL RETURNS


residual_returns = pd.DataFrame(
    index=strategy_returns.index
)

for strategy in strategy_returns.columns:

    results = run_factor_regression(
        strategy_returns[strategy],
        factor_returns,
        hac_lags=3,
    )

    residual_returns[
        strategy
    ] = results.resid



# 19. RESIDUAL VOLATILITY


residual_volatility = (
    residual_returns.std()
    * np.sqrt(12)
)

print(
    "\nResidual volatility:"
)

print(
    residual_volatility.round(4)
)



# 20. ALPHA COMPARISON


alpha_table = (
    attribution_summary[
        [
            "Alpha",
            "Alpha_tstat",
            "Residual_Volatility",
            "R_squared",
            "Observations",
        ]
    ]
)

print(
    "\nAlpha comparison:"
)

print(
    alpha_table.round(4)
)



# 21. SAVE OUTPUTS


factor_returns.to_csv(
    "task3_factor_returns.csv"
)

strategy_returns.to_csv(
    "task3_strategy_returns.csv"
)

attribution_summary.to_csv(
    "task3_attribution_summary.csv"
)

residual_returns.to_csv(
    "task3_residual_returns.csv"
)

print(
    "\nTask 3 outputs saved."
)

print("\nFactor diagnostics:")

print(
    factor_returns.describe().round(4)
)

print("\nFactor correlation:")

print(
    factor_returns.corr().round(4)
)

print("\nFactor standard deviations:")

print(
    factor_returns.std().round(6)
)
print("\nRaw factor samples:")

print(
    factor_returns[
        ["MKT", "SMB", "HML", "MOM"]
    ].tail(20)
)