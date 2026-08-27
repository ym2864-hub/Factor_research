import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Task 4: Performance Decomposition and Robustness Validation
# Provisional selected strategy: Raw Composite 50/50


DATA_PATH = "panel_with_liquidity.csv"
OUTPUT_DIR = Path("task4_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# 1. LOAD + DATA QUALITY

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

required = [
    "Date", "Ticker", "Return", "PE", "MarketCap",
    "Sector", "DollarVolume"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Match Task 3 data-quality rule.
df["Return_clean"] = df["Return"].copy()
invalid = (
    ~np.isfinite(df["Return"])
    | (df["Return"] <= -1)
    | (df["Return"] > 5)
)
df.loc[invalid, "Return_clean"] = np.nan

print("Data shape:", df.shape)
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("Stocks:", df["Ticker"].nunique())
print("Invalid returns removed:", int(invalid.sum()))
print("Liquidity proxy: DollarVolume")


# 2. SIGNAL CONSTRUCTION

def winsorize_cross_section(s, lower=0.01, upper=0.99):
    valid = s.dropna()
    if len(valid) < 5:
        return s
    lo = valid.quantile(lower)
    hi = valid.quantile(upper)
    return s.clip(lo, hi)


def make_signals(
    data,
    momentum_window=12,
    skip_months=1,
    winsor_rule="none",
    missing_rule="both_required",
):
    x = data.copy()
    x = x.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Value = earnings yield proxy.
    x["value_raw"] = np.where(
        (x["PE"].notna()) & (x["PE"] > 0),
        1.0 / x["PE"],
        np.nan,
    )

    # 12-1 style momentum by default.
    def momentum_for_stock(r):
        return (
            (1.0 + r)
            .rolling(momentum_window, min_periods=momentum_window)
            .apply(np.prod, raw=True)
            .shift(skip_months)
            - 1.0
        )

    x["momentum_raw"] = (
        x.groupby("Ticker")["Return_clean"]
        .transform(momentum_for_stock)
    )

    # Optional cross-sectional winsorization.
    if winsor_rule == "1_99":
        lo, hi = 0.01, 0.99
    elif winsor_rule == "2.5_97.5":
        lo, hi = 0.025, 0.975
    else:
        lo, hi = None, None

    if lo is not None:
        x["value_raw"] = (
            x.groupby("Date")["value_raw"]
            .transform(lambda s: winsorize_cross_section(s, lo, hi))
        )
        x["momentum_raw"] = (
            x.groupby("Date")["momentum_raw"]
            .transform(lambda s: winsorize_cross_section(s, lo, hi))
        )

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

    x["value_z"] = x.groupby("Date")["value_raw"].transform(zscore)
    x["momentum_z"] = x.groupby("Date")["momentum_raw"].transform(zscore)

    if missing_rule == "zero_fill":
        x["value_z"] = x["value_z"].fillna(0.0)
        x["momentum_z"] = x["momentum_z"].fillna(0.0)
        x["composite"] = 0.5 * x["value_z"] + 0.5 * x["momentum_z"]
    else:
        valid = x["value_z"].notna() & x["momentum_z"].notna()
        x["composite"] = np.nan
        x.loc[valid, "composite"] = (
            0.5 * x.loc[valid, "value_z"]
            + 0.5 * x.loc[valid, "momentum_z"]
        )

    return x

# Baseline parameters.
df_sig = make_signals(df)


# 3. PORTFOLIO ENGINE

def select_top_weights(
    data,
    score_col="composite",
    top_fraction=0.20,
    universe_mask=None,
    liquidity_weighted=False,
    rebalance_months=1,
):
    x = data.copy()
    if universe_mask is not None:
        x = x.loc[universe_mask(x)].copy()

    x["selected"] = False
    x["target_weight"] = 0.0

    # Rebalance only on every Nth month.
    dates = pd.Series(sorted(x["Date"].dropna().unique()))
    rebalance_dates = set(dates.iloc[::rebalance_months].tolist())

    for date, g in x.groupby("Date", sort=True):
        if date not in rebalance_dates:
            continue
        valid = g[score_col].notna()
        if valid.sum() == 0:
            continue
        n = max(1, int(np.ceil(valid.sum() * top_fraction)))
        chosen = g.loc[valid].nlargest(n, score_col).index
        x.loc[chosen, "selected"] = True

        if liquidity_weighted:
            liq = x.loc[chosen, "DollarVolume"].clip(lower=0).fillna(0)
            if liq.sum() > 0:
                x.loc[chosen, "target_weight"] = liq / liq.sum()
            else:
                x.loc[chosen, "target_weight"] = 1.0 / len(chosen)
        else:
            x.loc[chosen, "target_weight"] = 1.0 / len(chosen)

    weights = (
        x.pivot(index="Date", columns="Ticker", values="target_weight")
        .fillna(0.0)
        .sort_index()
    )

    # Hold the latest target until the next rebalance.
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)
    return weights


def portfolio_returns(data, weights):
    ret = (
        data.pivot(index="Date", columns="Ticker", values="Return_clean")
        .sort_index()
    )
    w = weights.reindex(index=ret.index, columns=ret.columns, fill_value=0.0)
    # Signal at t -> return during t+1.
    w = w.shift(1).fillna(0.0)
    return (ret * w).sum(axis=1).rename("Return")


def performance_stats(r):
    r = r.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
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


def run_strategy(
    data,
    momentum_window=12,
    skip_months=1,
    rebalance_months=1,
    top_fraction=0.20,
    winsor_rule="none",
    missing_rule="both_required",
    universe_mask=None,
    liquidity_weighted=False,
):
    sig = make_signals(
        data,
        momentum_window=momentum_window,
        skip_months=skip_months,
        winsor_rule=winsor_rule,
        missing_rule=missing_rule,
    )
    weights = select_top_weights(
        sig,
        top_fraction=top_fraction,
        universe_mask=universe_mask,
        liquidity_weighted=liquidity_weighted,
        rebalance_months=rebalance_months,
    )
    returns = portfolio_returns(sig, weights)
    return returns, weights, sig


# 4. BASELINE STRATEGY

baseline_returns, baseline_weights, baseline_sig = run_strategy(df)

print("\nBaseline performance:")
print(performance_stats(baseline_returns).round(4))


# 5. SUBPERIOD DECOMPOSITION

subperiods = {
    "2019-2020": ("2019-01-01", "2020-12-31"),
    "2021-2022": ("2021-01-01", "2022-12-31"),
    "2023-2024": ("2023-01-01", "2024-12-31"),
}

subperiod_rows = []
for name, (start, end) in subperiods.items():
    r = baseline_returns.loc[start:end]
    s = performance_stats(r)
    s["Period"] = name
    subperiod_rows.append(s)

subperiod_table = pd.DataFrame(subperiod_rows).set_index("Period")
print("\nSubperiod performance:")
print(subperiod_table.round(4))
subperiod_table.to_csv(OUTPUT_DIR / "subperiod_performance.csv")


# 6. MARKET REGIME DECOMPOSITION

market_proxy = (
    df.groupby("Date")["Return_clean"]
    .mean()
    .sort_index()
)
rolling_vol = market_proxy.rolling(12).std() * np.sqrt(12)
rolling_ret = (1 + market_proxy).rolling(12).apply(np.prod, raw=True) - 1

regime = pd.DataFrame({
    "MarketReturn": market_proxy,
    "RollingVol": rolling_vol,
    "Rolling12MReturn": rolling_ret,
})
vol_median = regime["RollingVol"].median()
regime["VolRegime"] = np.where(
    regime["RollingVol"] >= vol_median,
    "High Vol",
    "Low Vol",
)
regime["TrendRegime"] = np.where(
    regime["Rolling12MReturn"] >= 0,
    "Bull",
    "Bear",
)

regime_rows = []
for col in ["VolRegime", "TrendRegime"]:
    for label, idx in regime.groupby(col).groups.items():
        r = baseline_returns.reindex(idx).dropna()
        s = performance_stats(r)
        s["RegimeType"] = col
        s["Regime"] = label
        regime_rows.append(s)

regime_table = pd.DataFrame(regime_rows).set_index(["RegimeType", "Regime"])
print("\nRegime performance:")
print(regime_table.round(4))
regime_table.to_csv(OUTPUT_DIR / "regime_performance.csv")


# 7. QUINTILE DECOMPOSITION

q_data = baseline_sig.copy()
q_data["score_rank"] = q_data.groupby("Date")["composite"].rank(pct=True, method="first")
q_data["Quintile"] = pd.cut(
    q_data["score_rank"],
    bins=[0, .2, .4, .6, .8, 1.0],
    labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    include_lowest=True,
)
q_data["next_return"] = q_data.groupby("Ticker")["Return_clean"].shift(-1)
q_returns = q_data.groupby(["Date", "Quintile"], observed=True)["next_return"].mean().unstack()
q_summary = pd.DataFrame({
    "Average Monthly Return": q_returns.mean(),
    "Annualized Return Approx": q_returns.mean() * 12,
})
q_summary.loc["Q5-Q1"] = [
    q_summary.loc["Q5", "Average Monthly Return"] - q_summary.loc["Q1", "Average Monthly Return"],
    q_summary.loc["Q5", "Annualized Return Approx"] - q_summary.loc["Q1", "Annualized Return Approx"],
]
print("\nQuintile decomposition:")
print(q_summary.round(4))
q_summary.to_csv(OUTPUT_DIR / "quintile_decomposition.csv")




# 8. SECTOR DECOMPOSITION

weights_long = (
    baseline_weights
    .stack()
    .rename("Weight")
    .reset_index()
)

weights_long.columns = ["Date", "Ticker", "Weight"]

sector_map = (
    baseline_sig[
        ["Date", "Ticker", "Sector"]
    ]
    .drop_duplicates()
)

sector_weights = weights_long.merge(
    sector_map,
    on=["Date", "Ticker"],
    how="left"
)

# ------------------------------------------------------------
# Normalize portfolio weights within each date
# so that total portfolio weight = 1
# ------------------------------------------------------------


# Normalize only dates with non-zero portfolio weights
daily_weight_sum = (
    sector_weights
    .groupby("Date")["Weight"]
    .transform("sum")
)

valid_weight = daily_weight_sum > 0

sector_weights = sector_weights.loc[
    valid_weight
].copy()

sector_weights["Weight"] = (
    sector_weights["Weight"]
    / daily_weight_sum.loc[
        valid_weight
    ]
)
# Aggregate stock weights into sector weights
sector_exposure = (
    sector_weights
    .groupby(["Date", "Sector"])["Weight"]
    .sum()
    .unstack(fill_value=0)
)

# Check that sector weights sum to 1 each month
sector_weight_check = sector_exposure.sum(axis=1)

print("\nSector weight check:")
print(sector_weight_check.describe().round(6))

print("\nAverage sector exposure:")
print(
    sector_exposure
    .mean()
    .sort_values(ascending=False)
    .round(4)
)
sector_return = baseline_sig.pivot(index="Date", columns="Ticker", values="Return_clean")
sector_contribution = []
for date, g in sector_weights.groupby("Date"):
    r = sector_return.loc[date] if date in sector_return.index else pd.Series(dtype=float)
    for sector, sg in g.groupby("Sector"):
        rr = r.reindex(sg["Ticker"]).values
        ww = sg["Weight"].values
        contribution = np.nansum(ww * rr)
        sector_contribution.append([date, sector, contribution])

sector_contribution = pd.DataFrame(
    sector_contribution,
    columns=["Date", "Sector", "Contribution"]
)
sector_summary = sector_contribution.groupby("Sector")["Contribution"].agg(
    AverageMonthlyContribution="mean",
    TotalContribution="sum",
).sort_values("TotalContribution", ascending=False)

print("\nAverage sector exposure:")
print(sector_exposure.mean().sort_values(ascending=False).round(4))
print("\nSector contribution:")
print(sector_summary.round(4))
sector_exposure.to_csv(OUTPUT_DIR / "sector_exposure.csv")
sector_summary.to_csv(OUTPUT_DIR / "sector_contribution.csv")


# 9. HOLDING-PERIOD DECOMPOSITION

# Forward returns after signal formation. This is a diagnostic, not a
# tradable overlapping portfolio return series.
holding_rows = []
for h in [1, 3, 6, 12]:
    z = baseline_sig[["Date", "Ticker", "composite"]].copy()
    z["rank"] = z.groupby("Date")["composite"].rank(pct=True, method="first")
    z = z[z["rank"] >= 0.80].copy()
    future = (
        baseline_sig.set_index(["Ticker", "Date"])["Return_clean"]
        .sort_index()
    )
    vals = []
    for _, row in z.iterrows():
        t = row["Date"]
        ticker = row["Ticker"]
        dates = future.loc[ticker].index if ticker in future.index.get_level_values(0) else []
        if len(dates) == 0:
            continue
        try:
            pos = dates.get_loc(t)
            if isinstance(pos, slice) or pos + h >= len(dates):
                continue
            r = future.loc[ticker].iloc[pos + 1: pos + h + 1]
            if len(r) == h and r.notna().all():
                vals.append((1 + r).prod() - 1)
        except Exception:
            continue
    holding_rows.append({
        "HoldingMonths": h,
        "AverageForwardReturn": np.mean(vals) if vals else np.nan,
        "Observations": len(vals),
    })
holding_table = pd.DataFrame(holding_rows).set_index("HoldingMonths")
print("\nHolding-period decomposition:")
print(holding_table.round(4))
holding_table.to_csv(OUTPUT_DIR / "holding_period_decomposition.csv")


# 10. PARAMETER SENSITIVITY

parameter_rows = []
for window in [6, 9, 12, 18]:
    for skip in [1, 2]:
        r, _, _ = run_strategy(df, momentum_window=window, skip_months=skip)
        s = performance_stats(r)
        parameter_rows.append({
            "MomentumWindow": window,
            "SkipMonths": skip,
            **s.to_dict(),
        })

for rebalance in [1, 3]:
    r, _, _ = run_strategy(df, rebalance_months=rebalance)
    s = performance_stats(r)
    parameter_rows.append({
        "MomentumWindow": 12,
        "SkipMonths": 1,
        "RebalanceMonths": rebalance,
        **s.to_dict(),
    })

for q in [0.10, 0.20, 0.30]:
    r, _, _ = run_strategy(df, top_fraction=q)
    s = performance_stats(r)
    parameter_rows.append({
        "MomentumWindow": 12,
        "SkipMonths": 1,
        "TopFraction": q,
        **s.to_dict(),
    })

for wr in ["none", "1_99", "2.5_97.5"]:
    r, _, _ = run_strategy(df, winsor_rule=wr)
    s = performance_stats(r)
    parameter_rows.append({
        "MomentumWindow": 12,
        "SkipMonths": 1,
        "WinsorRule": wr,
        **s.to_dict(),
    })

for mr in ["both_required", "zero_fill"]:
    r, _, _ = run_strategy(df, missing_rule=mr)
    s = performance_stats(r)
    parameter_rows.append({
        "MomentumWindow": 12,
        "SkipMonths": 1,
        "MissingRule": mr,
        **s.to_dict(),
    })

parameter_table = pd.DataFrame(parameter_rows)
print("\nParameter sensitivity:")
print(parameter_table.round(4))
parameter_table.to_csv(OUTPUT_DIR / "parameter_sensitivity.csv", index=False)


# 11. UNIVERSE ROBUSTNESS

def large_cap_mask(x):
    threshold = x.groupby("Date")["MarketCap"].transform("median")
    return x["MarketCap"] >= threshold


def top_liquidity_mask(x):
    threshold = x.groupby("Date")["DollarVolume"].transform(lambda s: s.quantile(.70))
    return x["DollarVolume"] >= threshold


def sector_filtered_mask(x):
    # Exclude the two smallest sectors by average stock count in the universe.
    counts = x.groupby("Sector")["Ticker"].nunique().sort_values()
    excluded = set(counts.head(2).index)
    return ~x["Sector"].isin(excluded)

universe_tests = {
    "Full Universe": None,
    "Large Cap": large_cap_mask,
    "Top 30pct Liquidity": top_liquidity_mask,
    "Sector Filtered": sector_filtered_mask,
}

universe_rows = []
for name, mask_fn in universe_tests.items():
    r, _, _ = run_strategy(df, universe_mask=mask_fn)
    s = performance_stats(r)
    s["Universe"] = name
    universe_rows.append(s)

universe_table = pd.DataFrame(universe_rows).set_index("Universe")
print("\nUniverse robustness:")
print(universe_table.round(4))
universe_table.to_csv(OUTPUT_DIR / "universe_robustness.csv")


# 12. EQUAL-WEIGHT VS LIQUIDITY-WEIGHTED

eq_r, eq_w, _ = run_strategy(df, liquidity_weighted=False)
liq_r, liq_w, _ = run_strategy(df, liquidity_weighted=True)

liquidity_table = pd.DataFrame({
    "Equal Weight": performance_stats(eq_r),
    "Liquidity Weighted": performance_stats(liq_r),
}).T
print("\nEqual-weight vs liquidity-weighted:")
print(liquidity_table.round(4))
liquidity_table.to_csv(OUTPUT_DIR / "liquidity_comparison.csv")


# 13. CAPACITY PROXY

# Illustrative capacity proxy: assume a strategy participates in at most
# 5% of the DollarVolume proxy on each holding. This is not a broker-grade
# capacity estimate; it is a diagnostic constraint.
selected = baseline_weights.stack().rename("Weight").reset_index()
selected.columns = ["Date", "Ticker", "Weight"]
liq = df[["Date", "Ticker", "DollarVolume"]].drop_duplicates()
cap = selected.merge(liq, on=["Date", "Ticker"], how="left")
cap["MaxNotional_5pct"] = 0.05 * cap["DollarVolume"]
cap["ImpliedAUM"] = np.where(
    cap["Weight"] > 0,
    cap["MaxNotional_5pct"] / cap["Weight"],
    np.nan,
)
capacity_summary = cap["ImpliedAUM"].replace([np.inf, -np.inf], np.nan).describe(
    percentiles=[.01, .05, .25, .50, .75, .95, .99]
)
print("\nCapacity proxy summary (5% DollarVolume participation):")
print(capacity_summary.round(2))
capacity_summary.to_csv(OUTPUT_DIR / "capacity_proxy.csv")


# 14. DRAWDOWN / WORST MONTHS

wealth = (1 + baseline_returns.fillna(0)).cumprod()
drawdown = wealth / wealth.cummax() - 1
worst_months = baseline_returns.nsmallest(10).to_frame("Return")
worst_months["Drawdown"] = drawdown.reindex(worst_months.index)

print("\nWorst months:")
print(worst_months.round(4))
worst_months.to_csv(OUTPUT_DIR / "worst_months.csv")


# 15. FAILURE CONDITIONS / FRAGILE ASSUMPTIONS

failure_conditions = pd.DataFrame({
    "Condition": [
        "Strong dependence on market exposure",
        "Strong dependence on momentum exposure",
        "Performance concentrated in small or illiquid names",
        "Results materially weaken under alternative lookbacks or skips",
        "Results materially weaken under quarterly rebalancing",
        "Results materially weaken after sector filtering",
        "Results depend on extreme or invalid observations",
        "Results depend on one short subperiod or one sector",
    ],
    "What_To_Check": [
        "Compare raw returns with Task 3 factor attribution.",
        "Compare momentum beta and momentum factor contribution.",
        "Compare full, large-cap, and top-liquidity universes.",
        "Inspect parameter_sensitivity.csv.",
        "Inspect monthly versus quarterly results.",
        "Inspect universe_robustness.csv.",
        "Review the invalid-return data-quality filter.",
        "Inspect subperiod, regime, and sector tables.",
    ],
})
failure_conditions.to_csv(OUTPUT_DIR / "failure_conditions.csv", index=False)


# 16. CHARTS

cum = (1 + baseline_returns.fillna(0)).cumprod()
plt.figure(figsize=(12, 5))
plt.plot(cum.index, cum.values)
plt.title("Task 4: Baseline Composite Cumulative Wealth")
plt.xlabel("Date")
plt.ylabel("Cumulative Wealth")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_cumulative_wealth.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(drawdown.index, drawdown.values)
plt.title("Task 4: Baseline Composite Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "baseline_drawdown.png", dpi=150)
plt.show()

print("\nTask 4 outputs saved to:", OUTPUT_DIR.resolve())
