"""
Section 3.6 - Protokol Backtesting dan Validasi.

Computes the three metrics named in the proposal for both policies over
identical backtest days, then runs a paired t-test (scipy.stats.ttest_rel)
on each metric at alpha=0.05, exactly as specified:

    "Algoritma dinyatakan superior jika menghasilkan tingkat stockout dan
     biaya holding yang secara statistik lebih rendah dari baseline
     secara bersamaan."
     -> superiority requires stockout rate AND holding cost to both be
        significantly lower, simultaneously (restock frequency is
        reported but is NOT part of the superiority criterion per the
        proposal's own wording - don't let a low-frequency win alone
        count as "algorithm wins").

Unchanged from the original Colab pipeline -- not implicated in either
diagnosed bug.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

import config


@dataclass
class BacktestMetrics:
    stockout_rate: float          # fraction of days with a stockout event
    mean_daily_holding_cost: float
    restock_frequency: float      # fraction of days a reorder was triggered
    total_stockout_amount: float


def compute_metrics(policy_result: pd.DataFrame, price_series: pd.Series,
                     holding_cost_rate: float = config.DAILY_HOLDING_COST_RATE) -> BacktestMetrics:
    """
    Daily holding cost is modeled as holding_cost_rate * inventory_value,
    where inventory_value = inventory_end * price_on_that_date. This rate
    is an explicit assumption (see config.py) - the proposal names
    "biaya holding" as a metric but never gives a formula for it. Any
    reviewer should be told this up front, not discover it in the code.
    """
    prices = price_series.reindex(policy_result.index)
    inventory_value = policy_result["inventory_end"] * prices
    daily_holding_cost = inventory_value * holding_cost_rate

    return BacktestMetrics(
        stockout_rate=float(policy_result["stockout"].mean()),
        mean_daily_holding_cost=float(daily_holding_cost.mean()),
        restock_frequency=float(policy_result["reorder_triggered"].mean()),
        total_stockout_amount=float(policy_result["stockout_amount"].sum()),
    )


def paired_comparison(proposed: pd.DataFrame, baseline: pd.DataFrame,
                       price_series: pd.Series,
                       holding_cost_rate: float = config.DAILY_HOLDING_COST_RATE,
                       alpha: float = config.SIGNIFICANCE_ALPHA) -> dict:
    """
    Runs day-aligned paired t-tests on the three metrics' DAILY series
    (not just the summary means) - a paired t-test needs paired
    observations, so this compares stockout indicator, daily holding
    cost, and reorder indicator day-by-day across both policies on their
    common date index.

    Returns a dict with per-metric summary stats, t-statistic, p-value,
    and a plain-language verdict against the proposal's stated
    superiority criterion.
    """
    common_dates = proposed.index.intersection(baseline.index)
    if len(common_dates) < 2:
        raise ValueError("Need at least 2 overlapping backtest days for a paired t-test.")

    prices = price_series.reindex(common_dates)

    prop_stockout = proposed.loc[common_dates, "stockout"].astype(float)
    base_stockout = baseline.loc[common_dates, "stockout"].astype(float)

    prop_holding = proposed.loc[common_dates, "inventory_end"] * prices * holding_cost_rate
    base_holding = baseline.loc[common_dates, "inventory_end"] * prices * holding_cost_rate

    prop_restock = proposed.loc[common_dates, "reorder_triggered"].astype(float)
    base_restock = baseline.loc[common_dates, "reorder_triggered"].astype(float)

    def _paired_test(a: pd.Series, b: pd.Series, lower_is_better: bool):
        t_stat, p_val = stats.ttest_rel(a, b)
        mean_diff = float(a.mean() - b.mean())
        significant = p_val < alpha
        proposed_better = significant and (mean_diff < 0) if lower_is_better else significant
        return {
            "proposed_mean": float(a.mean()),
            "baseline_mean": float(b.mean()),
            "mean_diff_proposed_minus_baseline": mean_diff,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_at_alpha": bool(significant),
            "proposed_significantly_lower": bool(proposed_better),
        }

    stockout_test = _paired_test(prop_stockout, base_stockout, lower_is_better=True)
    holding_test = _paired_test(prop_holding, base_holding, lower_is_better=True)
    restock_test = _paired_test(prop_restock, base_restock, lower_is_better=True)  # reported, not a superiority criterion

    superior = stockout_test["proposed_significantly_lower"] and holding_test["proposed_significantly_lower"]

    return {
        "n_days_compared": int(len(common_dates)),
        "stockout_rate": stockout_test,
        "holding_cost": holding_test,
        "restock_frequency": restock_test,
        "proposal_superiority_criterion_met": bool(superior),
        "verdict": (
            "Proposed algorithm is superior per the proposal's own criterion "
            "(stockout rate AND holding cost both significantly lower, alpha=0.05)."
            if superior else
            "Proposal's superiority criterion NOT met - at least one of "
            "{stockout rate, holding cost} is not significantly lower than baseline."
        ),
    }
