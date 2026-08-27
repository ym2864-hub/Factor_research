import numpy as np
import pandas as pd


def fixed_bps_cost(
    traded_dollars,
    bps=10.0,
):
    """
    Fixed transaction cost.

    Parameters
    ----------
    traded_dollars : float or pd.Series
        Absolute dollar amount traded.

    bps : float
        Transaction cost in basis points.

        10 bps = 0.10%

    Returns
    -------
    float or pd.Series
        Transaction cost in dollars.
    """

    return traded_dollars * bps / 10_000.0


def bid_ask_cost(
    traded_dollars,
    spread_bps=20.0,
):
    """
    Bid-ask spread proxy.

    We approximate the cost of crossing the spread
    using a fixed spread assumption.

    Parameters
    ----------
    traded_dollars : float or pd.Series
        Absolute dollar amount traded.

    spread_bps : float
        Assumed bid-ask spread in basis points.

    Returns
    -------
    float or pd.Series
        Estimated bid-ask cost in dollars.
    """

    return traded_dollars * spread_bps / 10_000.0


def volume_based_cost(
    traded_dollars,
    average_daily_volume,
    impact_bps=10.0,
):
    """
    Simple volume-based transaction cost proxy.

    The larger the trade relative to available market
    volume, the larger the estimated cost.

    Parameters
    ----------
    traded_dollars : float or pd.Series
        Dollar amount traded.

    average_daily_volume : float or pd.Series
        Average daily dollar trading volume.

    impact_bps : float
        Maximum/reference impact in basis points.

    Returns
    -------
    float or pd.Series
        Estimated volume-based transaction cost.
    """

    traded_dollars = pd.Series(
        traded_dollars
    )

    average_daily_volume = pd.Series(
        average_daily_volume
    )

    participation = (
        traded_dollars
        / average_daily_volume.replace(
            0,
            np.nan,
        )
    )

    participation = (
        participation
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    participation = participation.clip(
        lower=0.0,
        upper=1.0,
    )

    cost = (
        traded_dollars
        * impact_bps
        / 10_000.0
        * participation
    )

    return cost


def illiquidity_slippage(
    traded_dollars,
    liquidity_score,
    slippage_bps=20.0,
):
    """
    Additional slippage for less-liquid securities.

    liquidity_score should be between 0 and 1:

        1.0 = highly liquid
        0.0 = very illiquid

    The less liquid the stock, the larger the
    additional slippage.

    Parameters
    ----------
    traded_dollars : float or pd.Series
        Dollar amount traded.

    liquidity_score : float or pd.Series
        Liquidity score between 0 and 1.

    slippage_bps : float
        Maximum additional slippage in basis points.

    Returns
    -------
    float or pd.Series
        Additional slippage cost in dollars.
    """

    traded_dollars = pd.Series(
        traded_dollars
    )

    liquidity_score = pd.Series(
        liquidity_score
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    illiquidity = 1.0 - liquidity_score

    return (
        traded_dollars
        * slippage_bps
        / 10_000.0
        * illiquidity
    )


def total_transaction_cost(
    traded_dollars,
    average_daily_volume=None,
    liquidity_score=None,
    fixed_bps=10.0,
    spread_bps=20.0,
    impact_bps=10.0,
    slippage_bps=20.0,
):
    """
    Combine fixed, bid-ask, volume-based, and
    illiquidity/slippage costs.

    Returns
    -------
    dict
        Individual cost components and total cost.
    """

    fixed = fixed_bps_cost(
        traded_dollars,
        bps=fixed_bps,
    )

    spread = bid_ask_cost(
        traded_dollars,
        spread_bps=spread_bps,
    )

    if average_daily_volume is not None:
        volume = volume_based_cost(
            traded_dollars,
            average_daily_volume,
            impact_bps=impact_bps,
        )
    else:
        volume = pd.Series(
            0.0,
            index=pd.Series(
                traded_dollars
            ).index,
        )

    if liquidity_score is not None:
        slippage = illiquidity_slippage(
            traded_dollars,
            liquidity_score,
            slippage_bps=slippage_bps,
        )
    else:
        slippage = pd.Series(
            0.0,
            index=pd.Series(
                traded_dollars
            ).index,
        )

    total = (
        fixed
        + spread
        + volume
        + slippage
    )

    return {
        "fixed_bps_cost": fixed,
        "bid_ask_cost": spread,
        "volume_cost": volume,
        "illiquidity_slippage": slippage,
        "total_cost": total,
    }