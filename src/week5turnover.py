import pandas as pd
import numpy as np


def get_rebalance_dates(
    dates,
    frequency="monthly",
):
    """
    Generate rebalance dates from a time series index.

    Parameters
    ----------
    dates : pd.DatetimeIndex or array-like
        Trading dates.

    frequency : str
        "monthly" or "quarterly".

    Returns
    -------
    pd.DatetimeIndex
        Rebalance dates.

    Notes
    -----
    The rebalance date is the last available trading day
    of each month or quarter.
    """

    dates = pd.DatetimeIndex(dates)

    if len(dates) == 0:
        return pd.DatetimeIndex([])

    dates = dates.sort_values()

    if frequency not in {"monthly", "quarterly"}:
        raise ValueError(
            "frequency must be 'monthly' or 'quarterly'"
        )

    # Convert dates into a Series so that we can group
    # trading days by month or quarter.
    date_series = pd.Series(dates, index=dates)

    if frequency == "monthly":
        period = date_series.index.to_period("M")
    else:
        period = date_series.index.to_period("Q")

    # Take the last available trading day in each period.
    rebalance_dates = (
        date_series
        .groupby(period)
        .max()
    )

    return pd.DatetimeIndex(
        rebalance_dates.values
    )


def calculate_turnover(
    trades,
    portfolio_value,
):
    """
    Calculate one-way and two-way portfolio turnover.

    Parameters
    ----------
    trades : pd.DataFrame
        Dollar trades by ticker and date.

        Positive value = buy
        Negative value = sell

    portfolio_value : pd.Series
        Total portfolio value at each date.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:

        one_way_turnover
        two_way_turnover

    Definitions
    -----------
    One-way turnover:

        sum(abs(trades)) / portfolio_value

    Two-way turnover:

        2 * one-way turnover
    """

    # Make sure the indices are aligned.
    trades = trades.copy()

    portfolio_value = portfolio_value.reindex(
        trades.index
    )

    # Total dollar amount traded on each date.
    total_traded_dollars = trades.abs().sum(axis=1)

    # Avoid division by zero.
    one_way_turnover = pd.Series(
        0.0,
        index=trades.index,
        dtype=float,
    )

    valid = portfolio_value > 0

    one_way_turnover.loc[valid] = (
        total_traded_dollars.loc[valid]
        / portfolio_value.loc[valid]
    )

    # Two-way turnover counts both sides of the trade.
    two_way_turnover = (
        2.0 * one_way_turnover
    )

    return pd.DataFrame(
        {
            "one_way_turnover": one_way_turnover,
            "two_way_turnover": two_way_turnover,
        }
    )


def calculate_position_changes(
    weights,
):
    """
    Calculate position changes by ticker and date.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights by date and ticker.

    Returns
    -------
    pd.DataFrame
        Change in portfolio weight for each ticker.

    The first date has zero position change because
    there is no previous portfolio.
    """

    weights = weights.copy()

    weights = weights.sort_index()

    position_changes = weights.diff()

    # No previous portfolio on the first date.
    position_changes.iloc[0] = 0.0

    return position_changes


def summarize_turnover(
    turnover,
):
    """
    Calculate summary statistics for turnover.

    Parameters
    ----------
    turnover : pd.DataFrame
        Output from calculate_turnover().

    Returns
    -------
    pd.Series
        Mean one-way and two-way turnover.
    """

    return pd.Series(
        {
            "average_one_way_turnover": (
                turnover["one_way_turnover"].mean()
            ),
            "average_two_way_turnover": (
                turnover["two_way_turnover"].mean()
            ),
        }
    )


def filter_rebalance_frequency(
    target_weights,
    frequency="monthly",
):
    """
    Select monthly or quarterly rebalance dates
    from target-weight dates.
    """

    target_weights = target_weights.sort_index()

    if frequency == "monthly":
        return target_weights

    if frequency == "quarterly":
        quarter = target_weights.index.to_period("Q")

        selected_dates = (
            pd.Series(
                target_weights.index,
                index=target_weights.index,
            )
            .groupby(quarter)
            .max()
        )

        return target_weights.loc[
            pd.DatetimeIndex(selected_dates.values)
        ]

    raise ValueError(
        "frequency must be 'monthly' or 'quarterly'"
    )