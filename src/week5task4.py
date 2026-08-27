import pandas as pd
import numpy as np

from src.liquidity import (
    load_liquidity_panel,
    apply_liquidity_filter,
)

from src.task2_analysis import run_task2


INITIAL_CAPITAL = 1_000_000.0


def calculate_capacity(
    trades,
    liquidity_df,
    participation_rate,
):
    """
    Estimate the maximum portfolio capital that can be
    supported under a given market participation constraint.

    trades are dollar trades for a $1M portfolio.

    For each trade:

        required_participation =
            trade_dollars / DollarVolume

    If the maximum allowed participation is p:

        capacity =
            $1M * p / required_participation

    The strategy capacity is the minimum capacity
    across all trades.
    """

    # Convert trades from wide to long format.
    trades_long = (
        trades
        .copy()
        .reset_index()
        .rename(columns={"index": "Date"})
        .melt(
            id_vars="Date",
            var_name="Ticker",
            value_name="trade_dollars",
        )
    )

    trades_long["Date"] = pd.to_datetime(
        trades_long["Date"]
    )

    trades_long["trade_dollars"] = (
        trades_long["trade_dollars"].abs()
    )

    # Liquidity data.
    liquidity = liquidity_df[
        [
            "Date",
            "Ticker",
            "DollarVolume",
        ]
    ].copy()

    liquidity["Date"] = pd.to_datetime(
        liquidity["Date"]
    )

    # Match each trade with its market dollar volume.
    merged = trades_long.merge(
        liquidity,
        on=["Date", "Ticker"],
        how="left",
    )

    # Only consider actual trades with valid liquidity.
    active = merged[
        (merged["trade_dollars"] > 0)
        & merged["DollarVolume"].notna()
        & (merged["DollarVolume"] > 0)
    ].copy()

    # Required market participation for the $1M portfolio.
    active["required_participation"] = (
        active["trade_dollars"]
        / active["DollarVolume"]
    )

    # Maximum portfolio size supported by this trade.
    active["implied_capacity"] = (
        INITIAL_CAPITAL
        * participation_rate
        / active["required_participation"]
    )

    # The tightest trade determines strategy capacity.
    capacity = active["implied_capacity"].min()

    worst_idx = active[
        "implied_capacity"
    ].idxmin()

    worst_trade = active.loc[
        worst_idx
    ]

    return {
        "capacity": capacity,
        "worst_trade": worst_trade,
        "trade_details": active,
    }



def run_task4(
    panel_path="panel_with_liquidity.csv",
    participation_rates=(0.01, 0.05, 0.10),
):
    """
    Run Task 4 liquidity and capacity analysis.
    """

    
    # Load liquidity data
    

    panel = load_liquidity_panel(
        panel_path
    )

    # Apply the liquidity filter.
    filtered_panel = apply_liquidity_filter(
        panel
    )

    
    # Run Task 2 strategies
    

    task2 = run_task2(
        "panel.csv"
    )

    capacity_results = {}

    
    # Capacity for every strategy
    

    for strategy_name, strategy_data in (
        task2["results"].items()
    ):

        trades = strategy_data[
            "results"
        ]["trades"]

        strategy_capacity = {}

        for participation_rate in (
            participation_rates
        ):

            result = calculate_capacity(
                trades=trades,
                liquidity_df=filtered_panel,
                participation_rate=participation_rate,
            )

            strategy_capacity[
                participation_rate
            ] = result

        capacity_results[
            strategy_name
        ] = strategy_capacity

    return {
        "filtered_panel": filtered_panel,
        "capacity_results": capacity_results,
    }

def calculate_before_after_performance(
    panel_path="panel.csv",
    liquidity_path="panel_with_liquidity.csv",
    liquidity_threshold=100_000_000,
):
    """
    Compare the same strategy construction before and after
    applying the DollarVolume liquidity filter.
    """

    # ---------------------------------------------
    # Load original panel
    # ---------------------------------------------

    from src.strategies import (
        load_panel,
        prepare_strategies,
    )

    original = load_panel(
        panel_path
    )

    # ---------------------------------------------
    # Load liquidity panel
    # ---------------------------------------------

    liquidity = pd.read_csv(
        liquidity_path
    )

    liquidity["Date"] = pd.to_datetime(
        liquidity["Date"]
    )

    # ---------------------------------------------
    # Apply $100M liquidity filter
    # ---------------------------------------------

    liquid_keys = liquidity[
        liquidity["DollarVolume"].notna()
        & (liquidity["DollarVolume"] > 0)
        & (
            liquidity["DollarVolume"]
            >= liquidity_threshold
        )
    ][
        ["Date", "Ticker"]
    ].drop_duplicates()

    # ---------------------------------------------
    # Filter original panel using Date + Ticker
    # ---------------------------------------------

    filtered = original.merge(
        liquid_keys,
        on=["Date", "Ticker"],
        how="inner",
    )

    # ---------------------------------------------
    # Build strategies using the SAME
    # prepare_strategies() function
    # ---------------------------------------------

    original_strategies = prepare_strategies(
        original
    )

    filtered_strategies = prepare_strategies(
        filtered
    )

    # ---------------------------------------------
    # Calculate portfolio returns
    # ---------------------------------------------

    def portfolio_returns(
        data,
        weights,
    ):
        """
        Calculate monthly portfolio returns using
        weights and stock returns from the panel.
        """

        returns = (
            data
            .pivot(
                index="Date",
                columns="Ticker",
                values="Return",
            )
            .sort_index()
        )

        weights = weights.reindex(
            index=returns.index,
            columns=returns.columns,
        ).fillna(0.0)

        portfolio_returns = (
            returns * weights
        ).sum(axis=1)

        return portfolio_returns

    rows = []

    for strategy_name in [
        "value_weights",
        "momentum_weights",
        "combined_weights",
    ]:

        before_returns = portfolio_returns(
            original,
            original_strategies[
                strategy_name
            ],
        )

        after_returns = portfolio_returns(
            filtered,
            filtered_strategies[
                strategy_name
            ],
        )

        before_cumulative = (
            1 + before_returns
        ).prod() - 1

        after_cumulative = (
            1 + after_returns
        ).prod() - 1

        rows.append(
            {
                "strategy":
                    strategy_name.replace(
                        "_weights",
                        "",
                    ),
                "before_filter":
                    before_cumulative,
                "after_filter":
                    after_cumulative,
                "change":
                    after_cumulative
                    - before_cumulative,
            }
        )

    return pd.DataFrame(rows)

if __name__ == "__main__":

    
    # Task 4: Liquidity and Strategy Capacity
    

    output = run_task4()

    print("\n=== Task 4 Strategy Capacity ===")

    for strategy, rates in output["capacity_results"].items():

        print(f"\n{strategy}")

        for rate, result in rates.items():

            capacity = result["capacity"]
            worst = result["worst_trade"]

            print(
                f"Participation rate {rate:.0%}: "
                f"${capacity:,.0f}"
            )

            print(
                f"  Worst trade: "
                f"{worst['Date'].date()} "
                f"{worst['Ticker']}"
            )

            print(
                f"  Trade: "
                f"${worst['trade_dollars']:,.0f}"
            )

            print(
                f"  Dollar volume: "
                f"${worst['DollarVolume']:,.0f}"
            )

            print(
                f"  Required participation: "
                f"{worst['required_participation']:.2%}"
            )

    
    # Before vs After Liquidity Filter
    

    performance = calculate_before_after_performance()

    print(
        "\n=== Before vs After Liquidity Filter ==="
    )

    display = performance.copy()

    for column in [
        "before_filter",
        "after_filter",
        "change",
    ]:

        display[column] = display[column].map(
            lambda x: f"{x:.2%}"
        )

    print(
        display.to_string(index=False)
    )

    # Save results
    performance.to_csv(
        "task4_liquidity_comparison.csv",
        index=False,
    )

    print(
        "\nSaved: "
        "task4_liquidity_comparison.csv"
    )
