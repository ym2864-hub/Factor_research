import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Week 7 Task 1: Final Performance Decomposition
# Selected strategy: Raw Value-Momentum Composite 50/50
# 12-1 momentum | monthly rebalance | top 20% | equal weight


DATA_PATH = "panel_with_liquidity.csv"
OUTPUT_DIR = Path("task1_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# 1. LOAD + DATA QUALITY

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

required = ["Date", "Ticker", "Return", "PE", "Sector"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["Return_clean"] = df["Return"].copy()
invalid = (
    ~np.isfinite(df["Return"])
    | (df["Return"] <= -1)
    | (df["Return"] > 5)
)
df.loc[invalid, "Return_clean"] = np.nan


# 2. SIGNAL CONSTRUCTION

def zscore(s):
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index)
    if valid.sum() == 0:
        return out
    std = s.loc[valid].std()
    if pd.isna(std) or std == 0:
        out.loc[valid] = 0.0
    else:
        out.loc[valid] = (s.loc[valid] - s.loc[valid].mean()) / std
    return out

x = df.copy()
x["value_raw"] = np.where(
    x["PE"].notna() & (x["PE"] > 0),
    1.0 / x["PE"],
    np.nan,
)

# 12-1 momentum: 12-month cumulative return, shifted by 1 month.
def momentum_for_stock(r):
    return (
        (1.0 + r)
        .rolling(12, min_periods=12)
        .apply(np.prod, raw=True)
        .shift(1)
        - 1.0
    )

x["momentum_raw"] = x.groupby("Ticker")["Return_clean"].transform(momentum_for_stock)
x["value_z"] = x.groupby("Date")["value_raw"].transform(zscore)
x["momentum_z"] = x.groupby("Date")["momentum_raw"].transform(zscore)

valid = x["value_z"].notna() & x["momentum_z"].notna()
x["composite"] = np.nan
x.loc[valid, "composite"] = (
    0.5 * x.loc[valid, "value_z"]
    + 0.5 * x.loc[valid, "momentum_z"]
)


# 3. PORTFOLIO CONSTRUCTION

# IMPORTANT FIX:
# Do NOT forward-fill zero weights. At each monthly rebalance,
# unselected stocks must become zero. Then shift weights by one
# month so signal at t is used for return at t+1.
weights = pd.DataFrame(
    0.0,
    index=sorted(x["Date"].unique()),
    columns=sorted(x["Ticker"].unique()),
)

for date, g in x.groupby("Date", sort=True):
    valid_g = g[g["composite"].notna()]
    if valid_g.empty:
        continue
    n = max(1, int(np.ceil(len(valid_g) * 0.20)))
    chosen = valid_g.nlargest(n, "composite")["Ticker"]
    weights.loc[date, chosen] = 1.0 / len(chosen)

returns = x.pivot(index="Date", columns="Ticker", values="Return_clean").sort_index()
weights = weights.reindex(index=returns.index, columns=returns.columns, fill_value=0.0)
effective_weights = weights.shift(1).fillna(0.0)

baseline_returns = (returns * effective_weights).sum(axis=1).rename("Return")


# 4. BASELINE CHECK

def performance_stats(r):
    r = r.dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return pd.Series({
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": dd.min(),
        "Observations": len(r),
    })

baseline = performance_stats(baseline_returns)
print("\n=== BASELINE ===")
print(baseline.round(4))
baseline.to_csv(OUTPUT_DIR / "baseline_performance.csv")


# 5. SECTOR DECOMPOSITION

sector_map = x[["Date", "Ticker", "Sector"]].drop_duplicates()
sector_rows = []

for date in returns.index:
    w = effective_weights.loc[date]
    r = returns.loc[date]
    smap = sector_map[sector_map["Date"] == date].set_index("Ticker")["Sector"]
    tmp = pd.DataFrame({"Weight": w, "Return": r}).join(smap.rename("Sector"), how="left")
    tmp = tmp[tmp["Weight"] != 0]
    for sector, g in tmp.groupby("Sector"):
        sector_rows.append({
            "Date": date,
            "Sector": sector,
            "Contribution": (g["Weight"] * g["Return"]).sum(),
            "Exposure": g["Weight"].sum(),
        })

sector_monthly = pd.DataFrame(sector_rows)
sector_summary = (
    sector_monthly.groupby("Sector")["Contribution"]
    .agg(AverageMonthlyContribution="mean", TotalContribution="sum")
    .sort_values("TotalContribution", ascending=False)
)
sector_summary["ShareOfTotalContribution"] = (
    sector_summary["TotalContribution"] / sector_summary["TotalContribution"].sum()
)
sector_summary.to_csv(OUTPUT_DIR / "sector_contribution.csv")
sector_monthly.to_csv(OUTPUT_DIR / "sector_contribution_by_month.csv", index=False)


# 6. QUINTILE DECOMPOSITION

q = x.copy()
q["score_rank"] = q.groupby("Date")["composite"].rank(pct=True, method="first")
q["Quintile"] = pd.cut(
    q["score_rank"],
    bins=[0, .2, .4, .6, .8, 1.0],
    labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    include_lowest=True,
)
q["next_return"] = q.groupby("Ticker")["Return_clean"].shift(-1)
q_returns = q.groupby(["Date", "Quintile"], observed=True)["next_return"].mean().unstack()
q_summary = pd.DataFrame({
    "Average Monthly Return": q_returns.mean(),
    "Annualized Return Approx": q_returns.mean() * 12,
})
q_summary.loc["Q5-Q1"] = [
    q_summary.loc["Q5", "Average Monthly Return"] - q_summary.loc["Q1", "Average Monthly Return"],
    q_summary.loc["Q5", "Annualized Return Approx"] - q_summary.loc["Q1", "Annualized Return Approx"],
]
q_summary.to_csv(OUTPUT_DIR / "quintile_decomposition.csv")


# 7. HOLDING-PERIOD DECOMPOSITION

# Diagnostic forward returns, NOT a tradable overlapping portfolio series.
holding_rows = []
rank_data = x[["Date", "Ticker", "composite"]].copy()
rank_data["rank"] = rank_data.groupby("Date")["composite"].rank(pct=True, method="first")
top_names = rank_data[rank_data["rank"] >= 0.80][["Date", "Ticker"]].copy()

for h in [1, 3, 6, 12]:
    future_h = x.copy()
    future_h["ForwardReturn"] = future_h.groupby("Ticker")["Return_clean"].transform(
        lambda s: (1 + s).rolling(h, min_periods=h).apply(np.prod, raw=True).shift(-h)
    )
    merged = top_names.merge(
        future_h[["Date", "Ticker", "ForwardReturn"]],
        on=["Date", "Ticker"],
        how="left",
    )
    vals = merged["ForwardReturn"].dropna() - 1.0
    holding_rows.append({
        "HoldingMonths": h,
        "AverageForwardReturn": vals.mean(),
        "Observations": len(vals),
    })

holding = pd.DataFrame(holding_rows).set_index("HoldingMonths")
holding.to_csv(OUTPUT_DIR / "holding_period_decomposition.csv")

# 8. MARKET REGIME DECOMPOSITION

market_proxy = x.groupby("Date")["Return_clean"].mean().sort_index()
rolling_vol = market_proxy.rolling(12).std() * np.sqrt(12)
rolling_ret = (1 + market_proxy).rolling(12).apply(np.prod, raw=True) - 1

regime = pd.DataFrame({
    "MarketReturn": market_proxy,
    "RollingVol": rolling_vol,
    "Rolling12MReturn": rolling_ret,
})

vol_median = regime["RollingVol"].median()
regime["VolRegime"] = np.where(regime["RollingVol"] >= vol_median, "High Vol", "Low Vol")
regime["TrendRegime"] = np.where(regime["Rolling12MReturn"] >= 0, "Bull", "Bear")

regime_rows = []
for regime_type in ["VolRegime", "TrendRegime"]:
    for label, idx in regime.groupby(regime_type).groups.items():
        r = baseline_returns.reindex(idx).dropna()
        s = performance_stats(r)
        regime_rows.append({
            "RegimeType": regime_type,
            "Regime": label,
            **s.to_dict(),
        })

regime_table = pd.DataFrame(regime_rows)
regime_table.to_csv(OUTPUT_DIR / "regime_performance.csv", index=False)


# 9. DRAWDOWN

wealth = (1 + baseline_returns.fillna(0)).cumprod()
drawdown = wealth / wealth.cummax() - 1
drawdown.rename("Drawdown").to_csv(OUTPUT_DIR / "drawdown_series.csv")


# 10. DURABILITY: SECTOR CONTRIBUTION BY SUBPERIOD

sector_monthly["Subperiod"] = pd.cut(
    sector_monthly["Date"],
    bins=[
        pd.Timestamp("2018-12-31"),
        pd.Timestamp("2020-12-31"),
        pd.Timestamp("2022-12-31"),
        pd.Timestamp("2024-12-31"),
    ],
    labels=["2019-2020", "2021-2022", "2023-2024"],
)

sector_durability = (
    sector_monthly.groupby(["Sector", "Subperiod"], observed=True)["Contribution"]
    .sum()
    .unstack(fill_value=0)
)

subperiod_cols = ["2019-2020", "2021-2022", "2023-2024"]
sector_durability["Positive_All_Subperiods"] = (
    sector_durability[subperiod_cols] > 0
).all(axis=1)
sector_durability.to_csv(OUTPUT_DIR / "sector_contribution_by_subperiod.csv")


# 11. FINAL CONTRIBUTOR TABLE

trend = regime_table[
    regime_table["RegimeType"] == "TrendRegime"
].set_index("Regime")

strongest_sector = sector_summary["TotalContribution"].idxmax()
weakest_sector = sector_summary["TotalContribution"].idxmin()

strongest_regime = trend["Sharpe"].idxmax()
weakest_regime = trend["Sharpe"].idxmin()

# Check whether the strongest sector has positive contribution
# across all tested subperiods.
strongest_sector_durable = sector_durability.loc[
    strongest_sector, "Positive_All_Subperiods"
]

if strongest_sector_durable:
    sector_interpretation = (
        f"{strongest_sector} is the largest contributor and "
        f"has positive contribution across all tested subperiods, "
        f"suggesting a relatively durable source of returns."
    )
else:
    sector_interpretation = (
        f"{strongest_sector} is the largest contributor, but its "
        f"contribution is not positive across all tested subperiods, "
        f"suggesting some period dependence."
    )

# Q5-Q1 spread
q5_q1_spread = q_summary.loc[
    "Q5-Q1", "Annualized Return Approx"
]

# 12-month diagnostic forward return
holding_12m = holding.loc[
    12, "AverageForwardReturn"
]

contributor_table = pd.DataFrame([
    {
        "Dimension": "Sector",
        "Strongest": strongest_sector,
        "Weakest_or_Risk": weakest_sector,
        "Interpretation": sector_interpretation,
    },
    {
        "Dimension": "Quintile",
        "Strongest": "Q5",
        "Weakest_or_Risk": "Weak Q5-Q1 separation",
        "Interpretation": (
            f"Q5-Q1 annualized spread = {q5_q1_spread:.2%}; "
            f"weak cross-sectional separation."
        ),
    },
    {
        "Dimension": "Holding Period",
        "Strongest": "12M",
        "Weakest_or_Risk": "Short horizon",
        "Interpretation": (
            f"12M average forward return = {holding_12m:.2%}; "
            f"diagnostic only."
        ),
    },
    {
        "Dimension": "Market Regime",
        "Strongest": strongest_regime,
        "Weakest_or_Risk": weakest_regime,
        "Interpretation": (
            f"Bull Sharpe = {trend.loc['Bull', 'Sharpe']:.2f}; "
            f"Bear Sharpe = {trend.loc['Bear', 'Sharpe']:.2f}; "
            f"performance is regime-dependent."
        ),
    },
    {
        "Dimension": "Drawdown",
        "Strongest": "N/A",
        "Weakest_or_Risk": "Maximum drawdown",
        "Interpretation": (
            f"Maximum drawdown = {drawdown.min():.2%}; "
            f"key downside risk."
        ),
    },
])

contributor_table.to_csv(
    OUTPUT_DIR / "final_contributor_table.csv",
    index=False
)
contributor_table.to_csv(OUTPUT_DIR / "final_contributor_table.csv", index=False)


# 12. FINAL CHARTS

plt.figure(figsize=(10, 6))
sector_summary["TotalContribution"].sort_values().plot(kind="barh")
plt.title("Figure 1. Return Contribution by Sector")
plt.xlabel("Cumulative Contribution")
plt.ylabel("Sector")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_01_sector_contribution.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 5))
q_summary.loc[["Q1", "Q2", "Q3", "Q4", "Q5"], "Average Monthly Return"].plot(kind="bar")
plt.title("Figure 2. Forward Return by Composite Score Quintile")
plt.xlabel("Quintile")
plt.ylabel("Average Monthly Return")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_02_quintile_returns.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(holding.index, holding["AverageForwardReturn"], marker="o")
plt.title("Figure 3. Diagnostic Forward Return by Holding Period")
plt.xlabel("Holding Period (months)")
plt.ylabel("Average Forward Return")
plt.xticks([1, 3, 6, 12])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_03_holding_period.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
trend.loc[["Bull", "Bear"], "Annualized Return"].plot(kind="bar")
plt.title("Figure 4. Performance Across Market Trend Regimes")
plt.xlabel("Market Regime")
plt.ylabel("Annualized Return")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_04_market_regime.png", dpi=180)
plt.close()

plt.figure(figsize=(11, 5))
plt.plot(drawdown.index, drawdown.values)
plt.title("Figure 5. Corrected Baseline Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_05_drawdown.png", dpi=180)
plt.close()


# 13. PRINT FINAL RESULTS

print("\n================ WEEK 7 TASK 1 ================")
print("\nBaseline:")
print(baseline.round(4))
print("\nSector:")
print(sector_summary.round(4))
print("\nQuintile:")
print(q_summary.round(4))
print("\nHolding period:")
print(holding.round(4))
print("\nMarket regime:")
print(regime_table.round(4))
print("\nMaximum drawdown:", round(drawdown.min(), 4))
print("\nSector durability:")
print(sector_durability.round(4))
print("\nFinal contributor table:")
print(contributor_table.to_string(index=False))
print("\nOutputs saved to:", OUTPUT_DIR.resolve())
