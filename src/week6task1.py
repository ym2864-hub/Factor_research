import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# 1. LOAD DATA


DATA_PATH = "panel_with_liquidity.csv"

df = pd.read_csv(DATA_PATH)

# Basic formatting
df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values(["Ticker", "Date"])
      .reset_index(drop=True)
)

print("Data shape:", df.shape)
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("Number of stocks:", df["Ticker"].nunique())



# 2. DATA CHECKS


required_columns = [
    "Date",
    "Ticker",
    "Return",
    "PE",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nMissing values:")
print(df[required_columns].isna().sum())



# 3. VALUE FACTOR


def calculate_value_score(df):

    df = df.copy()

    df["value_score"] = np.nan

    valid_pe = (
        df["PE"].notna()
        & (df["PE"] > 0)
    )

    df.loc[valid_pe, "value_score"] = (
        1.0 / df.loc[valid_pe, "PE"]
    )

    return df


df = calculate_value_score(df)



# 4. 12-1 MONTH MOMENTUM


def calculate_momentum_score(df):

    df = df.copy()

    df = (
        df.sort_values(["Ticker", "Date"])
          .reset_index(drop=True)
    )

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

    return df


df = calculate_momentum_score(df)



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
        series.loc[valid] - series.loc[valid].mean()
    ) / std

    return result

df["value_z"] = (
    df.groupby("Date")["value_score"]
      .transform(cross_sectional_zscore)
)

df["momentum_z"] = (
    df.groupby("Date")["momentum_score"]
      .transform(cross_sectional_zscore)
)



# 6. COMPOSITE FACTOR


# Equal-weight combination
df["composite_50_50"] = np.nan

valid = (
    df["value_score"].notna()
    & df["momentum_score"].notna()
)

df.loc[valid, "composite_50_50"] = (
    0.5 * df.loc[valid, "value_z"]
    + 0.5 * df.loc[valid, "momentum_z"]
)


# Value-heavy combination
df["composite_70_30"] = np.nan

df.loc[valid, "composite_70_30"] = (
    0.7 * df.loc[valid, "value_z"]
    + 0.3 * df.loc[valid, "momentum_z"]
)


# Momentum-heavy combination
df["composite_30_70"] = np.nan

df.loc[valid, "composite_30_70"] = (
    0.3 * df.loc[valid, "value_z"]
    + 0.7 * df.loc[valid, "momentum_z"]
)



# 7. SIGNAL TABLE


signal_table = df[
    [
        "Date",
        "Ticker",
        "PE",
        "value_score",
        "momentum_score",
        "value_z",
        "momentum_z",
        "composite_50_50",
        "composite_70_30",
        "composite_30_70",
    ]
].copy()

print("\nSignal table:")
print(signal_table.head())



# 8. TOP QUINTILE SELECTION


def select_top_quintile(
    df,
    score_column,
    top_fraction=0.20
):

    def select_date_group(group):

        valid_scores = group[score_column].notna()

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
                    n_valid * top_fraction
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



# 9. TARGET WEIGHTS


def make_target_weights(
    df,
    score_column,
    top_fraction=0.20
):

    temp = df.copy()

    temp["selected"] = (
        select_top_quintile(
            temp,
            score_column,
            top_fraction
        )
    )

    temp["target_weight"] = 0.0

    selected_count = (
        temp.groupby("Date")["selected"]
            .transform("sum")
    )

    valid_selection = (
        temp["selected"]
        & (selected_count > 0)
    )

    temp.loc[
        valid_selection,
        "target_weight"
    ] = (
        1.0
        / selected_count.loc[valid_selection]
    )

    weights = (
        temp.pivot(
            index="Date",
            columns="Ticker",
            values="target_weight"
        )
        .fillna(0.0)
        .sort_index()
    )

    return weights



# 10. CREATE FIVE STRATEGIES


score_columns = {

    "Value": "value_z",

    "Momentum": "momentum_z",

    "Composite_50_50": "composite_50_50",

    "Composite_70_30": "composite_70_30",

    "Composite_30_70": "composite_30_70",
}


weights = {}

for strategy_name, score_column in score_columns.items():

    weights[strategy_name] = (
        make_target_weights(
            df,
            score_column
        )
    )


print("\nStrategies created:")
print(list(weights.keys()))



# 11. CONVERT RETURNS TO MATRIX


returns = (
    df.pivot(
        index="Date",
        columns="Ticker",
        values="Return"
    )
    .sort_index()
)



# 12. CALCULATE PORTFOLIO RETURNS


def calculate_portfolio_returns(
    weights,
    returns
):

    # IMPORTANT:
    # Shift weights by one month.
    #
    # This prevents look-ahead bias:
    # signal at month t
    # -> portfolio held during month t+1

    lagged_weights = (
        weights
        .reindex(returns.index)
        .fillna(0.0)
        .shift(1)
    )

    # Align tickers
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


portfolio_returns = {}

for strategy_name in weights:

    portfolio_returns[strategy_name] = (
        calculate_portfolio_returns(
            weights[strategy_name],
            returns
        )
    )


portfolio_returns = pd.DataFrame(
    portfolio_returns
)

print("\nPortfolio returns:")
print(portfolio_returns.head())



# 13. PERFORMANCE STATISTICS


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
        returns.std() * np.sqrt(12)
    )

    if annualized_volatility > 0:

        sharpe = (
            annualized_return
            / annualized_volatility
        )

    else:
        sharpe = np.nan

    cumulative = (
        1 + returns
    ).cumprod()

    running_max = cumulative.cummax()

    drawdown = (
        cumulative / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    return {
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Observations": len(returns),
    }


performance_table = pd.DataFrame(
    {
        strategy: performance_stats(
            portfolio_returns[strategy]
        )
        for strategy in portfolio_returns.columns
    }
).T


print("\nPerformance comparison:")
print(
    performance_table.round(4)
)



# 14. CUMULATIVE PERFORMANCE


cumulative_returns = (
    1 + portfolio_returns
).cumprod()


plt.figure(figsize=(12, 6))

for strategy in cumulative_returns.columns:

    plt.plot(
        cumulative_returns.index,
        cumulative_returns[strategy],
        label=strategy
    )

plt.title(
    "Value, Momentum, and Composite Factor Performance"
)

plt.xlabel("Date")
plt.ylabel("Cumulative Wealth")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()



# 15. DRAWDOWN


drawdowns = pd.DataFrame(index=cumulative_returns.index)

for strategy in cumulative_returns.columns:

    running_max = (
        cumulative_returns[strategy]
        .cummax()
    )

    drawdowns[strategy] = (
        cumulative_returns[strategy]
        / running_max
        - 1
    )


plt.figure(figsize=(12, 6))

for strategy in drawdowns.columns:

    plt.plot(
        drawdowns.index,
        drawdowns[strategy],
        label=strategy
    )

plt.title("Strategy Drawdowns")

plt.xlabel("Date")
plt.ylabel("Drawdown")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()



# 16. FACTOR CORRELATION


factor_scores = df[
    [
        "value_z",
        "momentum_z",
    ]
].dropna()

factor_correlation = (
    factor_scores
    .corr()
)

print("\nValue / Momentum correlation:")
print(
    factor_correlation.round(4)
)



# 17. MONTHLY CROSS-SECTIONAL CORRELATION


def monthly_factor_correlation(group):

    return group["value_z"].corr(
        group["momentum_z"]
    )


monthly_correlations = (
    df.dropna(
        subset=[
            "value_z",
            "momentum_z"
        ]
    )
    .groupby("Date")
    .apply(
        monthly_factor_correlation
    )
)

print("\nAverage monthly Value-Momentum correlation:")

print(
    monthly_correlations.mean()
)



# 18. WEIGHTING SCHEME COMPARISON


comparison = performance_table[
    [
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe",
        "Max Drawdown",
    ]
].copy()

print("\nWeighting scheme comparison:")
print(
    comparison.round(4)
)



# 19. SAVE OUTPUTS


signal_table.to_csv(
    "task1_composite_signal_table.csv",
    index=False
)

performance_table.to_csv(
    "task1_performance_comparison.csv"
)

portfolio_returns.to_csv(
    "task1_portfolio_returns.csv"
)

cumulative_returns.to_csv(
    "task1_cumulative_returns.csv"
)

drawdowns.to_csv(
    "task1_drawdowns.csv"
)

print("\nTask 1 outputs saved.")