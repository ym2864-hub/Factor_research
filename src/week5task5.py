from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.task2_analysis import run_task2
from src.task3_analysis import run_task3



# Configuration


OUTPUT_DIR = Path("charts")
OUTPUT_DIR.mkdir(exist_ok=True)

STRATEGIES = [
    "Value_monthly",
    "Value_quarterly",
    "Momentum_monthly",
    "Momentum_quarterly",
    "Combined_monthly",
    "Combined_quarterly",
]



# Helper functions


def cumulative_returns(returns):
    """
    Convert periodic returns into cumulative returns.

    Example:
        returns = [0.01, -0.02, 0.03]

    becomes:
        (1.01 * 0.98 * 1.03) - 1
    """
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def calculate_drawdown(returns):
    """
    Calculate drawdown from a return series.
    """
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = wealth.cummax()

    drawdown = wealth / running_max - 1.0

    return drawdown



def calculate_rolling_sharpe(returns, window=12):
    """
    Calculate rolling annualized Sharpe ratio.

    Since the strategies are monthly,
    a 12-period window corresponds approximately
    to one year.
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    rolling_std = rolling_std.replace(0, float("nan"))

    sharpe = (
        rolling_mean / rolling_std
    ) * (12 ** 0.5)

    return sharpe.astype(float)

def get_return_series(task3_output, strategy):
    """
    Extract gross return series from Task 3 output.
    """
    results = task3_output[strategy]["results"]

    returns = results["returns"].copy()

    returns.index = pd.to_datetime(returns.index)

    return returns


def get_net_return_series(task3_output, strategy):
    """
    Construct net returns after transaction costs.

    Task 3 stores:
        gross returns
        total transaction costs
        portfolio value
    """

    result = task3_output[strategy]

    results = result["results"]
    costs = result["costs"]

    gross_returns = results["returns"].copy()
    diagnostics = results["diagnostics"]

    gross_returns.index = pd.to_datetime(gross_returns.index)

    portfolio_value = diagnostics["portfolio_value"].copy()
    portfolio_value.index = pd.to_datetime(portfolio_value.index)

    total_cost = costs["total_cost"].sum(axis=1)
    total_cost.index = pd.to_datetime(total_cost.index)

    # Cost as a fraction of portfolio value.
    cost_return = (
        total_cost / portfolio_value
    )

    cost_return = cost_return.reindex(
        gross_returns.index
    ).fillna(0.0)

    net_returns = gross_returns - cost_return

    return net_returns



# 1. Cumulative Returns


def plot_cumulative_returns(task3_output):
    """
    Plot gross cumulative returns for all strategies.
    """

    plt.figure(figsize=(12, 7))

    for strategy in STRATEGIES:

        returns = get_return_series(
            task3_output,
            strategy,
        )

        cumulative = cumulative_returns(returns)

        plt.plot(
            cumulative.index,
            cumulative.values,
            label=strategy,
        )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title("Cumulative Gross Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")

    plt.legend()

    plt.tight_layout()

    path = OUTPUT_DIR / "cumulative_returns.png"

    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved: {path}")



# 2. Drawdowns


def plot_drawdowns(task3_output):
    """
    Plot drawdowns for all strategies.
    """

    plt.figure(figsize=(12, 7))

    for strategy in STRATEGIES:

        returns = get_return_series(
            task3_output,
            strategy,
        )

        drawdown = calculate_drawdown(
            returns
        )

        plt.plot(
            drawdown.index,
            drawdown.values,
            label=strategy,
        )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title("Strategy Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")

    plt.legend()

    plt.tight_layout()

    path = OUTPUT_DIR / "drawdowns.png"

    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved: {path}")



# 3. Rolling Sharpe


def plot_rolling_sharpe(task3_output):
    """
    Plot 12-period rolling annualized Sharpe ratios.
    """

    plt.figure(figsize=(12, 7))

    for strategy in STRATEGIES:

        returns = get_return_series(
            task3_output,
            strategy,
        )

        sharpe = calculate_rolling_sharpe(
            returns,
            window=12,
        )

        plt.plot(
            sharpe.index,
            sharpe.values,
            label=strategy,
        )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.title(
        "12-Period Rolling Annualized Sharpe Ratio"
    )

    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe")

    plt.legend()

    plt.tight_layout()

    path = OUTPUT_DIR / "rolling_sharpe.png"

    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved: {path}")



# 4. Turnover


def create_turnover_summary(task2_output):
    """
    Extract average one-way and two-way turnover
    from Task 2 output.
    """

    rows = []

    for strategy in STRATEGIES:

        strategy_name, frequency = strategy.split("_")

        result = task2_output["summary"]

        row = result[
            (result["strategy"] == strategy_name)
            & (result["frequency"] == frequency)
        ]

        if len(row) == 0:
            continue

        row = row.iloc[0]

        rows.append(
            {
                "strategy": strategy_name,
                "frequency": frequency,
                "one_way_turnover":
                    row["average_one_way_turnover"],
                "two_way_turnover":
                    row["average_two_way_turnover"],
            }
        )

    return pd.DataFrame(rows)


def plot_turnover(task2_output):
    """
    Plot average one-way turnover by strategy
    and rebalance frequency.
    """

    summary = create_turnover_summary(
        task2_output
    )

    plt.figure(figsize=(10, 6))

    labels = []
    values = []

    for _, row in summary.iterrows():

        label = (
            f"{row['strategy']}\n"
            f"{row['frequency']}"
        )

        labels.append(label)
        values.append(row["one_way_turnover"])

    plt.bar(
        labels,
        values,
    )

    plt.title("Average One-Way Turnover")
    plt.ylabel("Turnover")
    plt.xlabel("Strategy / Frequency")

    plt.xticks(rotation=30)

    plt.tight_layout()

    path = OUTPUT_DIR / "turnover.png"

    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved: {path}")



# 5. Gross vs Net / Cost Drag


def create_cost_summary(task3_output):
    """
    Create gross return, net return,
    and cost-drag summary.
    """

    rows = []

    for strategy in STRATEGIES:

        gross_returns = get_return_series(
            task3_output,
            strategy,
        )

        net_returns = get_net_return_series(
            task3_output,
            strategy,
        )

        gross_cumulative = (
            (1 + gross_returns.fillna(0)).prod()
            - 1
        )

        net_cumulative = (
            (1 + net_returns.fillna(0)).prod()
            - 1
        )

        cost_drag = (
            gross_cumulative
            - net_cumulative
        )

        strategy_name, frequency = (
            strategy.split("_")
        )

        rows.append(
            {
                "strategy": strategy_name,
                "frequency": frequency,
                "gross_return": gross_cumulative,
                "net_return": net_cumulative,
                "cost_drag": cost_drag,
            }
        )

    return pd.DataFrame(rows)


def plot_cost_drag(task3_output):
    """
    Plot gross versus net cumulative returns.
    """

    summary = create_cost_summary(
        task3_output
    )

    plt.figure(figsize=(10, 6))

    labels = []

    gross_values = []
    net_values = []

    for _, row in summary.iterrows():

        labels.append(
            f"{row['strategy']}\n"
            f"{row['frequency']}"
        )

        gross_values.append(
            row["gross_return"]
        )

        net_values.append(
            row["net_return"]
        )

    x = range(len(labels))

    width = 0.35

    gross_x = [
        i - width / 2
        for i in x
    ]

    net_x = [
        i + width / 2
        for i in x
    ]

    plt.bar(
        gross_x,
        gross_values,
        width=width,
        label="Gross",
    )

    plt.bar(
        net_x,
        net_values,
        width=width,
        label="Net",
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.xticks(
        list(x),
        labels,
        rotation=30,
    )

    plt.title(
        "Gross vs Net Cumulative Returns"
    )

    plt.ylabel("Cumulative Return")

    plt.legend()

    plt.tight_layout()

    path = OUTPUT_DIR / "cost_drag.png"

    plt.savefig(path, dpi=200)

    plt.close()

    print(f"Saved: {path}")



# 6. Summary CSV


def save_summary(task3_output, task2_output):
    """
    Save the main Task 5 summary table.
    """

    cost_summary = create_cost_summary(
        task3_output
    )

    turnover_summary = create_turnover_summary(
        task2_output
    )

    summary = cost_summary.merge(
        turnover_summary,
        on=["strategy", "frequency"],
        how="left",
    )

    path = Path(
        "backtest_summary.csv"
    )

    summary.to_csv(
        path,
        index=False,
    )

    print(f"Saved: {path}")

    return summary



# Main


if __name__ == "__main__":

    print("\nRunning Task 5...\n")

    
    # Run Task 2 and Task 3
    

    print("Loading Task 2 results...")

    task2_output = run_task2(
        "panel.csv"
    )

    print("Loading Task 3 results...")

    task3_output = run_task3()

    
    # Generate charts
    

    print("\nGenerating charts...\n")

    plot_cumulative_returns(
        task3_output
    )

    plot_drawdowns(
        task3_output
    )

    plot_rolling_sharpe(
        task3_output
    )

    plot_turnover(
        task2_output
    )

    plot_cost_drag(
        task3_output
    )

    
    # Save summary
    

    print("\nCreating summary...\n")

    summary = save_summary(
        task3_output,
        task2_output,
    )

    print("\n=== Task 5 Summary ===")
    print(summary.to_string(index=False))

    print("\nTask 5 completed.")