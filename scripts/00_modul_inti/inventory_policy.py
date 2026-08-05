"""
Section 3.5 - Derivasi Parameter Safety Stock dan Reorder Point.
Section 2.6 - Metode Fixed Buffer Statis sebagai Baseline Perbandingan.

Implements, verbatim from the proposal:

    SS_t  = Z * sigma_d_t * sqrt(L)
    ROP_t = d_bar_t * L + SS_t
    Q_t   = target_inventory - I_t + SS_t     (only triggered when I_t <= ROP_t)
    target_inventory = P95 of simulated L-day demand distribution

Baseline:
    fixed buffer = BASELINE_BUFFER_PCT * rolling_30d_mean(volume)
    (proposal says "persentase tetap dari rata-rata volume transaksi harian
    30 hari terakhir" but does not name the percentage -> this is a
    documented assumption in config.py, not something taken from the text.)

Unchanged from the original Colab pipeline -- this module's logic already
matches the proposal formulas and was not implicated in either diagnosed
bug (unit bug, elasticity overflow bug).
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class PolicyDayResult:
    date: object
    inventory_start: float
    demand_realized: float
    reorder_triggered: bool
    reorder_qty: float
    stockout: bool
    stockout_amount: float
    inventory_end: float
    ss: float | None = None
    rop: float | None = None
    target_inventory: float | None = None
    buffer: float | None = None


def compute_ss_rop(sigma_d: float, d_bar: float, target_inventory: float,
                    z: float = config.Z_SCORE, lead_time: int = config.LEAD_TIME_DAYS):
    """Proposal eq. (3.5): SS_t, ROP_t. Returns (ss, rop)."""
    ss = z * sigma_d * np.sqrt(lead_time)
    rop = d_bar * lead_time + ss
    return ss, rop


def run_proposed_policy(realized_demand: pd.Series, mc_params: pd.DataFrame,
                         initial_inventory: float) -> pd.DataFrame:
    """
    Simulates the Monte-Carlo-driven dynamic SS/ROP policy day by day over
    the backtest period.

    realized_demand: actual observed daily volume during backtest (proxy
        for customer buy-order demand hitting CEX inventory, per proposal
        3.3 last sentence).
    mc_params: output of monte_carlo.walk_forward_daily_params, must cover
        the same date range as realized_demand (sigma_d, d_bar, p95_demand
        computed using only information available as of each date).

    IMPORTANT OPERATIONALIZATION NOTE (not fully specified in the proposal):
    the proposal defines SS/ROP/Q as formulas, but does not specify the
    day-to-day inventory bookkeeping loop (does a stockout roll over unmet
    demand to tomorrow? does restocked inventory arrive same-day given
    L=1?). This implementation makes the simplest choices consistent with
    L=1 (restock arrives same day it's triggered; unmet demand is lost,
    not backlogged) and documents them here explicitly rather than
    burying them in code. Change these assumptions if your committee
    wants a backlog model instead of a lost-sales model.
    """
    dates = realized_demand.index.intersection(mc_params.index)
    if len(dates) == 0:
        raise ValueError("No overlapping dates between realized_demand and mc_params.")

    inventory = initial_inventory
    rows = []
    for date in dates:
        sigma_d = mc_params.loc[date, "sigma_d"]
        d_bar = mc_params.loc[date, "d_bar"]
        target_inventory = mc_params.loc[date, "p95_demand"]
        ss, rop = compute_ss_rop(sigma_d, d_bar, target_inventory)

        inv_start = inventory
        reorder_triggered = inv_start <= rop
        reorder_qty = max(0.0, target_inventory - inv_start + ss) if reorder_triggered else 0.0
        inv_after_reorder = inv_start + reorder_qty

        demand_today = float(realized_demand.loc[date])
        stockout = demand_today > inv_after_reorder
        stockout_amount = max(0.0, demand_today - inv_after_reorder)
        inv_end = max(0.0, inv_after_reorder - demand_today)

        rows.append(PolicyDayResult(
            date=date, inventory_start=inv_start, demand_realized=demand_today,
            reorder_triggered=reorder_triggered, reorder_qty=reorder_qty,
            stockout=stockout, stockout_amount=stockout_amount, inventory_end=inv_end,
            ss=ss, rop=rop, target_inventory=target_inventory,
        ).__dict__)
        inventory = inv_end

    return pd.DataFrame(rows).set_index("date")


def run_fixed_buffer_policy(realized_demand: pd.Series, full_volume_history: pd.Series,
                             initial_inventory: float,
                             rolling_window: int = config.BASELINE_ROLLING_WINDOW_DAYS,
                             buffer_pct: float = config.BASELINE_BUFFER_PCT) -> pd.DataFrame:
    """
    Proposal 2.6 baseline: buffer = buffer_pct * mean(volume, last 30 days),
    with NO adjustment for current volatility conditions. Reorder trigger:
    inventory <= buffer -> restock up to buffer level.

    full_volume_history must include enough pre-backtest days to compute
    the first rolling window (i.e. pass the full series, not just the
    backtest slice); it's sliced internally to realized_demand's dates.
    """
    rolling_buffer = (full_volume_history.rolling(rolling_window).mean() * buffer_pct)
    dates = realized_demand.index.intersection(rolling_buffer.dropna().index)
    if len(dates) == 0:
        raise ValueError("No overlapping dates - check that full_volume_history "
                          "has enough pre-backtest history for the rolling window.")

    inventory = initial_inventory
    rows = []
    for date in dates:
        buffer = float(rolling_buffer.loc[date])

        inv_start = inventory
        reorder_triggered = inv_start <= buffer
        reorder_qty = max(0.0, buffer - inv_start) if reorder_triggered else 0.0
        inv_after_reorder = inv_start + reorder_qty

        demand_today = float(realized_demand.loc[date])
        stockout = demand_today > inv_after_reorder
        stockout_amount = max(0.0, demand_today - inv_after_reorder)
        inv_end = max(0.0, inv_after_reorder - demand_today)

        rows.append(PolicyDayResult(
            date=date, inventory_start=inv_start, demand_realized=demand_today,
            reorder_triggered=reorder_triggered, reorder_qty=reorder_qty,
            stockout=stockout, stockout_amount=stockout_amount, inventory_end=inv_end,
            buffer=buffer,
        ).__dict__)
        inventory = inv_end

    return pd.DataFrame(rows).set_index("date")
