import pandas as pd


def load_liquidity_panel(
    path="panel_with_liquidity.csv",
):
    """
    Load panel data containing DollarVolume.
    """

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    required_columns = {
        "Date",
        "Ticker",
        "DollarVolume",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    return df


def apply_liquidity_filter(
    df,
    min_dollar_volume=100_000_000,
):
    """
    Keep only stocks with sufficient
    dollar volume.

    Stocks with missing or non-positive
    dollar volume are considered
    non-tradable.
    """

    filtered = df[
        df["DollarVolume"].notna()
        & (df["DollarVolume"] > 0)
        & (
            df["DollarVolume"]
            >= min_dollar_volume
        )
    ].copy()

    return filtered


def liquidity_statistics(df):
    """
    Summarize liquidity by date.
    """

    stats = (
        df.groupby("Date")
        .agg(
            number_of_stocks=(
                "Ticker",
                "nunique",
            ),
            median_dollar_volume=(
                "DollarVolume",
                "median",
            ),
            total_dollar_volume=(
                "DollarVolume",
                "sum",
            ),
        )
    )

    return stats


def calculate_capacity(
    df,
    participation_rate=0.10,
):
    """
    Estimate capacity using a participation-rate
    assumption.

    A strategy can participate in at most
    participation_rate of each stock's
    dollar volume.
    """

    capacity = df.copy()

    capacity["stock_capacity"] = (
        capacity["DollarVolume"]
        * participation_rate
    )

    daily_capacity = (
        capacity.groupby("Date")
        ["stock_capacity"]
        .sum()
    )

    return daily_capacity


def summarize_capacity(
    df,
    participation_rates=(
        0.01,
        0.05,
        0.10,
    ),
):
    """
    Calculate capacity under several
    participation-rate assumptions.
    """

    rows = []

    for rate in participation_rates:

        daily_capacity = calculate_capacity(
            df,
            participation_rate=rate,
        )

        rows.append(
            {
                "participation_rate": rate,
                "median_capacity":
                    daily_capacity.median(),
                "minimum_capacity":
                    daily_capacity.min(),
                "maximum_capacity":
                    daily_capacity.max(),
            }
        )

    return pd.DataFrame(rows)

def calculate_stock_capacity(
    df,
    participation_rate=0.10,
):
    """
    Estimate the maximum dollar amount that can be
    traded in each stock under a participation-rate
    assumption.

    Example:
        DollarVolume = $100M
        participation_rate = 10%

        capacity = $10M
    """

    result = df[
        [
            "Date",
            "Ticker",
            "DollarVolume",
        ]
    ].copy()

    result["stock_capacity"] = (
        result["DollarVolume"]
        * participation_rate
    )

    return result


def capacity_summary(
    df,
    participation_rates=(
        0.01,
        0.05,
        0.10,
    ),
):
    """
    Summarize daily aggregate trading capacity.

    Participation rate represents the maximum fraction
    of market dollar volume that the strategy is assumed
    willing to trade.
    """

    rows = []

    for rate in participation_rates:

        stock_capacity = calculate_stock_capacity(
            df,
            participation_rate=rate,
        )

        daily_capacity = (
            stock_capacity
            .groupby("Date")
            ["stock_capacity"]
            .sum()
        )

        rows.append(
            {
                "participation_rate": rate,
                "median_daily_capacity":
                    daily_capacity.median(),
                "minimum_daily_capacity":
                    daily_capacity.min(),
                "maximum_daily_capacity":
                    daily_capacity.max(),
            }
        )

    return pd.DataFrame(rows)