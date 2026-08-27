import numpy as np
import pandas as pd

from src.backtest import BacktestEngine
from src.strategies import prepare_strategies
from src.turnover import filter_rebalance_frequency
from src.costs import (
    fixed_bps_cost,
    bid_ask_cost,
    volume_based_cost,
    illiquidity_slippage,
)


def load_panel_with_liquidity(path="panel_with_liquidity.csv"):
    """
    Load the Task 3 panel containing:
        Return
        MarketCap
        DollarVolume
        PE
        etc.
    """

    df = pd.read_csv(
        path,
        parse_dates=["Date"],
    )

    df = df.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    return df


def build_price_panel(df):
    """
    Reconstruct synthetic prices from monthly returns.

    Each stock starts at 100.
    """

    returns = (
        df.pivot(
            index="Date",
            columns="Ticker",
            values="Return",
        )
        .sort_index()
        .fillna(0.0)
    )

    prices = (
        100.0
        * (1.0 + returns)
        .cumprod()
    )

    return prices


def build_liquidity_panel(df):
    """
    Build a Date × Ticker DollarVolume panel.
    """

    dollar_volume = (
        df.pivot(
            index="Date",
            columns="Ticker",
            values="DollarVolume",
        )
        .sort_index()
    )

    return dollar_volume


def build_liquidity_score(df):
    """
    Construct a simple cross-sectional liquidity score.

    Larger DollarVolume = more liquid.

    The score is between 0 and 1.
    """

    dv = df[
        [
            "Date",
            "Ticker",
            "DollarVolume",
        ]
    ].copy()

    dv["liquidity_score"] = (
        dv.groupby("Date")[
            "DollarVolume"
        ]
        .rank(
            pct=True,
        )
    )

    liquidity_score = (
        dv.pivot(
            index="Date",
            columns="Ticker",
            values="liquidity_score",
        )
        .sort_index()
    )

    return liquidity_score


def run_gross_backtest(
    prices,
    target_weights,
    frequency,
    initial_capital=1_000_000.0,
):
    """
    Run the existing backtest engine without
    transaction costs.
    """

    selected_weights = filter_rebalance_frequency(
        target_weights,
        frequency=frequency,
    )

    selected_weights = selected_weights.reindex(
        columns=prices.columns,
        fill_value=0.0,
    )

    engine = BacktestEngine(
        prices=prices,
        target_weights=selected_weights,
        rebalance_dates=selected_weights.index,
        initial_capital=initial_capital,
    )

    results = engine.run()

    return results


def calculate_transaction_costs(
    trades,
    prices,
    dollar_volume,
    liquidity_score,
    portfolio_values,
    turnover_one_way,
    fixed_bps=10.0,
    spread_bps=20.0,
    impact_bps=10.0,
    slippage_bps=20.0,
):
    """
    Calculate transaction costs from actual backtest trades.

    The BacktestEngine already calculates one-way turnover
    using its own internally consistent dollar-trade logic.

    Therefore, we use:

        true traded dollars
        = one-way turnover × portfolio value

    We then allocate this total traded amount across
    tickers according to the relative trade sizes implied
    by the shares traded.

    This prevents the synthetic-price reconstruction from
    producing an incorrect total dollar volume.
    """

    
    # 1. Raw ticker-level trade sizes
    

    raw_dollar_trades = (
        trades.abs()
        * prices
    )

    raw_dollar_trades = (
        raw_dollar_trades
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    
    # 2. True total traded dollars
    #
    # BacktestEngine already knows the correct turnover.
    

    true_total_dollars = (
        turnover_one_way
        * portfolio_values
    )

    true_total_dollars = (
        true_total_dollars
        .fillna(0.0)
    )

    
    # 3. Scale ticker-level trades so that their sum
    #    exactly matches the engine's turnover.
    

    dollar_trades = pd.DataFrame(
        0.0,
        index=raw_dollar_trades.index,
        columns=raw_dollar_trades.columns,
    )

    for date in raw_dollar_trades.index:

        row = raw_dollar_trades.loc[date]

        raw_total = row.sum()

        target_total = true_total_dollars.loc[date]

        if raw_total > 0:

            dollar_trades.loc[date] = (
                row
                / raw_total
                * target_total
            )

    
    # 4. Fixed bps cost
    

    fixed_cost = fixed_bps_cost(
        dollar_trades,
        bps=fixed_bps,
    )

    
    # 5. Bid-ask spread proxy
    

    spread_cost = bid_ask_cost(
        dollar_trades,
        spread_bps=spread_bps,
    )

    
    # 6. Volume-based impact
    

    safe_volume = (
        dollar_volume
        .replace(
            0.0,
            np.nan,
        )
    )

    participation = (
        dollar_trades
        / safe_volume
    )

    participation = (
        participation
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    volume_cost = (
        dollar_trades
        * impact_bps
        / 10_000.0
        * participation
    )

    
    # 7. Illiquidity slippage
    

    score = liquidity_score.clip(
        lower=0.0,
        upper=1.0,
    )

    illiquidity = (
        1.0 - score
    )

    slippage_cost = (
        dollar_trades
        * slippage_bps
        / 10_000.0
        * illiquidity
    )

    
    # 8. Total
    

    total_cost = (
        fixed_cost
        + spread_cost
        + volume_cost
        + slippage_cost
    )

    return {
        "dollar_trades": dollar_trades,
        "fixed_cost": fixed_cost,
        "spread_cost": spread_cost,
        "volume_cost": volume_cost,
        "slippage_cost": slippage_cost,
        "total_cost": total_cost,
    }


def calculate_net_returns(
    gross_returns,
    total_cost,
    portfolio_values,
):
    """
    Convert dollar transaction costs into return drag.

    Cost drag:

        transaction cost
        / portfolio value before trading

    Net return:

        gross return - cost drag
    """

    cost_by_date = (
        total_cost.sum(axis=1)
    )

    cost_drag = (
        cost_by_date
        / portfolio_values
    )

    cost_drag = cost_drag.reindex(
        gross_returns.index
    ).fillna(0.0)

    net_returns = (
        gross_returns
        - cost_drag
    )

    return (
        cost_drag,
        net_returns,
    )


def run_task3(
    panel_path="panel_with_liquidity.csv",
):
    """
    Run Task 3 for:

        Value
        Momentum
        Combined

    under:

        Monthly
        Quarterly
    """

    df = load_panel_with_liquidity(
        panel_path
    )

    prices = build_price_panel(df)

    dollar_volume = (
        build_liquidity_panel(df)
    )

    liquidity_score = (
        build_liquidity_score(df)
    )

    strategies = prepare_strategies(
        df
    )

    strategy_weights = {
        "Value": strategies[
            "value_weights"
        ],
        "Momentum": strategies[
            "momentum_weights"
        ],
        "Combined": strategies[
            "combined_weights"
        ],
    }

    output = {}

    for strategy_name, weights in (
        strategy_weights.items()
    ):

        for frequency in [
            "monthly",
            "quarterly",
        ]:

            key = (
                f"{strategy_name}_"
                f"{frequency}"
            )

            results = run_gross_backtest(
                prices=prices,
                target_weights=weights,
                frequency=frequency,
            )

            
            costs = calculate_transaction_costs(
                trades=results["trades"],
                prices=prices,
                dollar_volume=dollar_volume,
                liquidity_score=liquidity_score,
                portfolio_values=results["diagnostics"]["portfolio_value"],
                turnover_one_way=results["diagnostics"]["turnover_one_way"],
            )

            cost_drag, net_returns = (
                calculate_net_returns(
                    gross_returns=results[
                        "returns"
                    ],
                    total_cost=costs[
                        "total_cost"
                    ],
                    portfolio_values=results[
                        "diagnostics"
                    ]["portfolio_value"],
                )
            )

            output[key] = {
                "results": results,
                "costs": costs,
                "cost_drag": cost_drag,
                "net_returns": net_returns,
            }

    return output


def cumulative_return(returns):
    """
    Calculate cumulative compounded return.
    """
    return (1.0 + returns).prod() - 1.0


def build_cost_sensitivity_table(output):
    """
    Compare gross performance with different
    transaction-cost assumptions.

    Cost models:

        1. Fixed bps
        2. Bid-ask proxy
        3. Volume-based impact
        4. Illiquidity slippage
        5. Full cost model
    """

    rows = []

    for key, result in output.items():

        strategy, frequency = key.rsplit(
            "_",
            1,
        )

        gross_returns = result[
            "results"
        ]["returns"]

        diagnostics = result[
            "results"
        ]["diagnostics"]

        portfolio_values = diagnostics[
            "portfolio_value"
        ]

        costs = result[
            "costs"
        ]

        
        # Cost components
        

        fixed_cost = (
            costs["fixed_cost"]
            .sum(axis=1)
        )

        spread_cost = (
            costs["spread_cost"]
            .sum(axis=1)
        )

        volume_cost = (
            costs["volume_cost"]
            .sum(axis=1)
        )

        slippage_cost = (
            costs["slippage_cost"]
            .sum(axis=1)
        )

        total_cost = (
            costs["total_cost"]
            .sum(axis=1)
        )

        
        # Convert dollar costs into return drag
        

        fixed_drag = (
            fixed_cost
            / portfolio_values
        ).reindex(
            gross_returns.index
        ).fillna(0.0)

        spread_drag = (
            spread_cost
            / portfolio_values
        ).reindex(
            gross_returns.index
        ).fillna(0.0)

        volume_drag = (
            volume_cost
            / portfolio_values
        ).reindex(
            gross_returns.index
        ).fillna(0.0)

        slippage_drag = (
            slippage_cost
            / portfolio_values
        ).reindex(
            gross_returns.index
        ).fillna(0.0)

        total_drag = (
            total_cost
            / portfolio_values
        ).reindex(
            gross_returns.index
        ).fillna(0.0)

        
        # Net return under each cost model
        

        fixed_net = (
            gross_returns
            - fixed_drag
        )

        spread_net = (
            gross_returns
            - spread_drag
        )

        volume_net = (
            gross_returns
            - volume_drag
        )

        slippage_net = (
            gross_returns
            - slippage_drag
        )

        full_net = (
            gross_returns
            - total_drag
        )

        
        # Store summary
        

        rows.append(
            {
                "strategy": strategy,
                "frequency": frequency,

                "gross_return":
                    cumulative_return(
                        gross_returns
                    ),

                "fixed_net_return":
                    cumulative_return(
                        fixed_net
                    ),

                "spread_net_return":
                    cumulative_return(
                        spread_net
                    ),

                "volume_net_return":
                    cumulative_return(
                        volume_net
                    ),

                "slippage_net_return":
                    cumulative_return(
                        slippage_net
                    ),

                "full_cost_net_return":
                    cumulative_return(
                        full_net
                    ),

                "full_cost_drag":
                    total_drag.sum(),

                "number_of_periods":
                    len(gross_returns),
            }
        )

    return pd.DataFrame(rows)


def print_cost_sensitivity(table):
    """
    Print the final Task 3 cost-sensitivity table.
    """

    display_table = table.copy()

    return_columns = [
        "gross_return",
        "fixed_net_return",
        "spread_net_return",
        "volume_net_return",
        "slippage_net_return",
        "full_cost_net_return",
        "full_cost_drag",
    ]

    for column in return_columns:

        display_table[column] = (
            display_table[column]
            .map(
                lambda x: f"{x:.2%}"
            )
        )

    print(
        "\n=== Task 3 Cost Sensitivity ==="
    )

    print(
        display_table.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    output = run_task3()

    
    # Original Task 3 summary
    

    print(
        "\n=== Task 3 Transaction Cost Summary ==="
    )

    for key, result in output.items():

        gross = result[
            "results"
        ]["returns"]

        net = result[
            "net_returns"
        ]

        cost_drag = result[
            "cost_drag"
        ]

        print(
            f"\n{key}"
        )

        print(
            f"Gross cumulative return: "
            f"{cumulative_return(gross):.4%}"
        )

        print(
            f"Net cumulative return:   "
            f"{cumulative_return(net):.4%}"
        )

        print(
            f"Total cost drag:          "
            f"{cost_drag.sum():.4%}"
        )

    
    # Cost sensitivity
    

    sensitivity = (
        build_cost_sensitivity_table(
            output
        )
    )

    print_cost_sensitivity(
        sensitivity
    )

    
    # Save final table
    

    sensitivity.to_csv(
        "task3_cost_sensitivity.csv",
        index=False,
    )

    print(
        "\nSaved:"
        " task3_cost_sensitivity.csv"
    )