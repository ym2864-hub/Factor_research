# Value–Momentum Equity Strategy Research

## Overview
This project studies a combined Value–Momentum equity strategy and evaluates its performance, risk, robustness, liquidity, capacity, and transaction costs.

The main research panel contains monthly observations from February 2018 through December 2024 across 607 stocks.

## Setup and Run

Clone the repository:

```bash
git clone https://github.com/ym2864-hub/Factor_research.git
cd Factor_research
```
## Data and Factors
Main inputs:
- Stock returns
- P/E ratios
- Market capitalization
- Sector classification
- Dollar trading volume

**Value:** `1 / PE`

**Momentum:** 12–1 month momentum, using the previous twelve months while excluding the most recent month.

**Combined:** 50% Value + 50% Momentum, followed by portfolio-weight normalization.

## Backtest Methodology
The backtest calculates target weights, compares them with current holdings, determines trades, and tracks holdings, cash, portfolio value, turnover, and returns.

To reduce look-ahead bias, signals use information available at the signal date and trades are executed on the next trading day.

## Baseline Performance

| Strategy | Annualized Return | Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Value | 9.22% | 22.23% | 0.4150 | -34.90% |
| Momentum | 14.01% | 19.19% | 0.7301 | -22.40% |
| **Composite 50/50** | **14.16%** | **17.30%** | **0.8185** | **-21.76%** |
| Composite 70/30 | 14.06% | 17.43% | 0.8066 | -22.38% |
| Composite 30/70 | 13.70% | 17.26% | 0.7939 | -21.86% |

The **50/50 Value–Momentum Composite** is the preferred baseline because it has the highest Sharpe ratio among the tested combinations.

## Robustness Checks

### Factor Weights
The 50/50 composite has the highest Sharpe ratio (0.819), versus 0.807 for 70/30 and 0.794 for 30/70. The differences are modest.

### Sector Neutralization
Sector neutralization reduces volatility from 17.30% to 16.36%, but also reduces return from 14.16% to 12.37% and Sharpe from 0.819 to 0.756. It is therefore not selected for the baseline.

### Subperiods
| Period | Annualized Return | Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| 2019–2020 | 51.28% | 55.75% | 0.9199 | -52.46% |
| 2021–2022 | -10.59% | 79.52% | -0.1332 | -75.06% |
| 2023–2024 | 75.13% | 76.83% | 0.9780 | -47.07% |

The strategy is sensitive to market conditions, with **2021–2022 as the clearest failure period**.

### Universe
| Universe | Annualized Return | Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Full Universe | 28.32% | 66.17% | 0.4280 | -75.06% |
| Large Cap | 40.86% | 64.21% | 0.6364 | -70.32% |
| Top 30% Liquidity | 19.56% | 94.31% | 0.2074 | -92.10% |
| Sector Filtered | 27.58% | 66.86% | 0.4125 | -76.27% |

The Large Cap universe has the strongest risk-adjusted performance.

### Signal Ranking
The Q5–Q1 spread is only **0.50% annually**, indicating weak separation between the highest- and lowest-ranked stocks.

## Factor Attribution
A four-factor regression uses market, size, value, and momentum factors.

Key results for the Raw Composite:
- Alpha: 3.56% per month
- Alpha t-statistic: 1.62
- Market beta: 1.01
- Momentum beta: 0.37
- Four-factor R²: 94.82%

The alpha t-statistic is below the conventional 5% significance threshold. The strategy should therefore **not be described as a pure alpha strategy**; much of its return variation is associated with market and momentum exposures.

## Transaction Costs
The cost model includes fixed basis-point costs, a bid-ask spread proxy, and volume-based costs.

| Strategy | Frequency | Gross Return | Net Return | Cost Drag |
|---|---|---:|---:|---:|
| Value | Monthly | 81.35% | 76.34% | 2.80% |
| Value | Quarterly | 78.05% | 75.00% | 1.75% |
| Momentum | Monthly | 117.36% | 89.48% | 13.83% |
| Momentum | Quarterly | 108.28% | 93.11% | 7.64% |
| Combined | Monthly | 142.96% | 112.13% | 13.69% |
| Combined | Quarterly | 151.55% | 133.65% | 7.50% |

Transaction costs have a much larger impact on high-turnover Momentum and Combined strategies. Quarterly rebalancing generally has lower cost drag than monthly rebalancing.

## Liquidity and Capacity
Liquidity is estimated using dollar trading volume, and stocks with insufficient volume are removed.

At a 1% participation rate:

| Strategy | Estimated Capacity |
|---|---:|
| Value Monthly | $87.4M |
| Value Quarterly | $78.6M |
| Momentum Monthly | $33.9M |
| Momentum Quarterly | $35.4M |
| Combined Monthly | $38.1M |
| Combined Quarterly | $51.2M |

Higher-turnover strategies generally have lower capacity because they require more trading.

After the liquidity filter, Combined cumulative return falls from 161.26% to 114.79%, while Value and Momentum remain relatively stable.

## Key Conclusions
1. The 50/50 Value–Momentum Composite is the preferred baseline.
2. Performance is sensitive to market regimes, especially the 2021–2022 failure period.
3. Sector neutralization lowers volatility but does not improve risk-adjusted performance.
4. The strategy has meaningful market and momentum exposures and is not a pure alpha strategy.
5. The small Q5–Q1 spread indicates weak ranking separation.
6. Transaction costs materially reduce performance, especially for high-turnover strategies.
7. Liquidity and capacity are important implementation constraints.

## Limitations
- Transaction costs are estimated rather than based on actual executed trades.
- The historical sample covers 2018–2024 and may not represent future conditions.
- The four-factor attribution uses only 70 monthly observations.
- The backtest does not establish out-of-sample performance.
- Capacity estimates depend on the assumed participation and cost model.

## Reproducibility
Run the notebooks in the intended order from a clean state. Before finalizing results:
1. Re-run critical backtests and robustness checks.
2. Check signal timing and look-ahead-bias controls.
3. Check transaction-cost assumptions.
4. Check annualization conventions.
5. Confirm report and presentation numbers match current outputs.

## Project Structure

```text
Factor_research/
├── README.md
├── .gitignore
├── data/
│   ├── panel_with_liquidity.csv
│   └── outputs/
│       ├── week5charts/
│       ├── week5task3_outputs/
│       ├── week6task1_outputs/
│       ├── week6task2_outputs/
│       ├── week6task3_outputs/
│       ├── week6task4_outputs/
│       ├── week6task5_outputs/
│       └── week7task1_outputs/
├── notebooks/
│   ├── week5_backtesting.ipynb
│   └── week6_Factor Combination and Attribution.ipynb
└── src/
    ├── week5costs.py
    ├── week5liquidity.py
    ├── week5task2.py
    ├── week5task3.py
    ├── week5task4.py
    ├── week5task5.py
    ├── week5turnover.py
    ├── week6task1.py
    ├── week6task2.py
    ├── week6task3.py
    ├── week6task4.py
    └── week7task1.py
