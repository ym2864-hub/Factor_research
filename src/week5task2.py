import numpy as np
import pandas as pd

from src.backtest import BacktestEngine
from src.strategies import (
    load_panel,
    prepare_strategies,
)
from src.turnover import (
    filter_rebalance_frequency,
)


def build_price_panel(df):
    """
    Convert monthly stock returns into a synthetic price panel.

    The backtest engine requires prices, while panel.csv
    contains monthly stock returns.

    Each stock starts at 100 and compounds its monthly returns.
    """

    returns = (
        df.pivot(
            index="Date",
            columns="Ticker",
            values="Return",
        )
        .sort_index()
    )

    # For the price reconstruction, missing monthly returns
    # are treated as 0 for that month.
    returns = returns.fillna(0.0)

    prices = (
        100.0
        * (1.0 + returns)
        .cumprod()
    )

    return prices


def run_strategy(
    prices,
    target_weights,
    frequency="monthly",
    initial_capital=1_000_000.0,
):
    """
    Run one strategy at one rebalance frequency.

    Parameters
    ----------
    prices : DataFrame
        Stock price panel.

    target_weights : DataFrame
        Strategy target weights by date and ticker.

    frequency : str
        "monthly" or "quarterly".

    Returns
    -------
    dict
        Backtest results plus turnover information.
    """

    # Select the appropriate rebalance dates.
    selected_weights = filter_rebalance_frequency(
        target_weights,
        frequency=frequency,
    )

    rebalance_dates = selected_weights.index

    # Keep only stocks available in the price panel.
    selected_weights = selected_weights.reindex(
        columns=prices.columns,
        fill_value=0.0,
    )

    engine = BacktestEngine(
        prices=prices,
        target_weights=selected_weights,
        rebalance_dates=rebalance_dates,
        initial_capital=initial_capital,
    )

    results = engine.run()

    return {
        "results": results,
        "target_weights": selected_weights,
        "rebalance_dates": rebalance_dates,
    }



def extract_turnover(results):
    """
    Extract turnover information from BacktestEngine results.

    BacktestEngine provides one-way turnover.
    Two-way turnover is defined as twice one-way turnover.
    """

    diagnostics = results["diagnostics"].copy()

    turnover = pd.DataFrame(
        index=diagnostics.index
    )

    # One-way turnover is already calculated
    # by BacktestEngine.
    turnover["one_way_turnover"] = (
        diagnostics["turnover_one_way"]
    )

    # Two-way turnover is twice one-way turnover.
    turnover["two_way_turnover"] = (
        2.0
        * turnover["one_way_turnover"]
    )

    return turnover


def calculate_position_changes(
    results,
    rebalance_dates,
):
    """
    Calculate position changes at execution dates.

    Position change is measured as the change in portfolio
    weight from the previous portfolio to the current
    portfolio.

    The first execution is measured relative to an all-cash
    portfolio, so the initial investment is included.
    """

    weights = results["weights"].copy()

    available_dates = [
        date
        for date in rebalance_dates
        if date in weights.index
    ]

    rebalance_weights = weights.loc[
        available_dates
    ].copy()

    # Previous portfolio is all cash.
    previous_weights = pd.Series(
        0.0,
        index=rebalance_weights.columns,
    )

    position_changes = []

    for date in rebalance_weights.index:

        current_weights = rebalance_weights.loc[date]

        change = (
            current_weights
            - previous_weights
        )

        position_changes.append(change)

        previous_weights = current_weights

    position_changes = pd.DataFrame(
        position_changes,
        index=rebalance_weights.index,
        columns=rebalance_weights.columns,
    )

    return position_changes


def summarize_strategy(
    strategy_name,
    frequency,
    results,
):
    """
    Produce one summary row for a strategy/frequency
    combination.
    """

    turnover = extract_turnover(
        results["results"]
    )

    traded_dates = turnover[
        turnover["one_way_turnover"] > 0
    ]

    if len(traded_dates) == 0:
        average_one_way = 0.0
        average_two_way = 0.0
    else:
        average_one_way = (
            traded_dates["one_way_turnover"]
            .mean()
        )

        average_two_way = (
            traded_dates["two_way_turnover"]
            .mean()
        )

    # Approximate number of rebalances per year.
    if frequency == "monthly":
        rebalances_per_year = 12
    elif frequency == "quarterly":
        rebalances_per_year = 4
    else:
        raise ValueError(
            "frequency must be monthly or quarterly"
        )

    # Annualized turnover based on average turnover
    # per rebalance.
    annualized_one_way = (
        average_one_way
        * rebalances_per_year
    )

    annualized_two_way = (
        average_two_way
        * rebalances_per_year
    )

    # Illustrative transaction-cost assumption:
    # 10 basis points per one-way traded dollar.
    transaction_cost_bps = 10

    annual_cost_drag = (
        annualized_one_way
        * transaction_cost_bps
        / 10000.0
    )

    return {
        "strategy": strategy_name,
        "frequency": frequency,
        "average_one_way_turnover": average_one_way,
        "average_two_way_turnover": average_two_way,
        "annualized_one_way_turnover": annualized_one_way,
        "annualized_two_way_turnover": annualized_two_way,
        "transaction_cost_bps": transaction_cost_bps,
        "estimated_annual_cost_drag": annual_cost_drag,
        "number_of_rebalances": len(
            results["rebalance_dates"]
        ),
    }

def run_task2(panel_path="panel.csv"):
    """
    Run all six Task 2 combinations:

        Value       Monthly
        Value       Quarterly
        Momentum    Monthly
        Momentum    Quarterly
        Combined    Monthly
        Combined    Quarterly

    Returns
    -------
    dict
        All backtest results and summary tables.
    """

    # 1. Load panel

    df = load_panel(panel_path)

    # 2. Build price panel

    prices = build_price_panel(df)

    # 3. Build strategies

    strategies = prepare_strategies(df)

    strategy_weights = {
        "Value": strategies["value_weights"],
        "Momentum": strategies["momentum_weights"],
        "Combined": strategies["combined_weights"],
    }

    # 4. Run six backtests

    all_results = {}

    summary_rows = []

    for strategy_name, weights in strategy_weights.items():

        for frequency in [
            "monthly",
            "quarterly",
        ]:

            result = run_strategy(
                prices=prices,
                target_weights=weights,
                frequency=frequency,
            )

            key = (
                f"{strategy_name}_{frequency}"
            )

            all_results[key] = result

            summary_rows.append(
                summarize_strategy(
                    strategy_name,
                    frequency,
                    result,
                )
            )

    # 5. Summary table

    summary = pd.DataFrame(
        summary_rows
    )

    return {
        "data": df,
        "prices": prices,
        "results": all_results,
        "summary": summary,
    }

if __name__ == "__main__":

    output = run_task2(
        panel_path="panel.csv"
    )

    print("\n=== Task 2 Turnover Summary ===")

    print(
        output["summary"].to_string(
            index=False
        )
    )

    
    # Position changes
    

    for key, result in output["results"].items():

        position_changes = calculate_position_changes(
            results=result["results"],
            rebalance_dates=result["rebalance_dates"],
        )

        print(
            f"\n=== Position Changes: {key} ==="
        )

        # Find the largest absolute position changes
        # for this particular strategy.
        abs_changes = (
            position_changes.abs()
            .stack()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        print(
            "Largest position changes:"
        )

        for (date, ticker), value in abs_changes.items():

            actual_change = (
                position_changes.loc[
                    date,
                    ticker,
                ]
            )

            print(
                f"{date.date()}  "
                f"{ticker:<8} "
                f"{actual_change:+.4%}"
            )
        
    # Save Task 2 outputs
    

    import os

    output_dir = "task2_output"
    os.makedirs(output_dir, exist_ok=True)

    # Save summary table
    output["summary"].to_csv(
        f"{output_dir}/turnover_summary.csv",
        index=False,
    )

    # Save turnover series and position changes
    for key, result in output["results"].items():

        
        # Turnover series
        

        turnover = extract_turnover(
            result["results"]
        )

        turnover.to_csv(
            f"{output_dir}/{key}_turnover.csv"
        )

        
        # Position changes
        

        position_changes = (
            calculate_position_changes(
                results=result["results"],
                rebalance_dates=result["rebalance_dates"],
            )
        )

        position_changes.to_csv(
            f"{output_dir}/{key}_position_changes.csv"
        )

    print(
        f"\nTask 2 outputs saved to: "
        f"{output_dir}/"
    )
