"""
Section 3.4 - Simulasi Monte Carlo (demand-scenario generator).

Implements the four numbered steps in the proposal exactly:
  1. Empirical log-return distribution from the calibration window
     (no distributional assumption; non-parametric bootstrap).
  2. For each of N=10,000 scenarios, bootstrap L sequential days of
     log-returns (sampling WITH replacement) -> N price paths of length L.
  3. Apply an empirically-estimated price-demand elasticity to convert
     each price path into a demand scenario.
  4. Extract sigma_d (std of simulated daily demand) and d_bar (mean of
     simulated daily demand) from the N scenarios.

Also implements the walk-forward re-run described in 3.4 final paragraph:
"Simulasi dijalankan ulang setiap hari perdagangan" - i.e. sigma_d and
d_bar are re-estimated daily using only data available up to that day.

BUG FIX vs the original Colab pipeline (crypto_inventory_mc_colab.ipynb)
=========================================================================
The original implementation was:

    demand_paths = historical_volume_mean * np.exp(elasticity * draws)

This is unbounded: as the walk-forward window expands to include
crisis-era returns (2022 includes the FTX collapse, a single-day BTC/ETH
move on the order of -20% to -45% log-return), `elasticity * draws` can
become large enough that `np.exp(...)` overflows to `inf`, or produces
finite-but-physically-absurd demand multipliers (thousands of times
baseline volume). This was diagnosed in
Draft_Artikel_Preliminer_Monte_Carlo_Inventori_Kripto_rev.docx (Bagian 5)
as the source of ETH's non-physical stockout metrics in the preliminary
run.

Fix: the exponent `elasticity * r` is clipped to
[-config.EXPONENT_CLIP, +config.EXPONENT_CLIP] before exponentiating.
This keeps demand strictly positive (as the log-linear form was designed
to do) AND bounded, at the cost of understating the true multiplier on
the most extreme historical days. That trade-off is a documented,
reportable modeling choice -- not hidden inside the function.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

import s1_config as config


@dataclass
class DemandScenarioOutput:
    sigma_d: float          # std dev of simulated L-day-ahead daily demand
    d_bar: float            # mean of simulated L-day-ahead daily demand
    p95_demand: float       # 95th percentile of simulated L-day cumulative demand
    scenarios: np.ndarray   # raw (N, L) simulated demand paths, kept for diagnostics
    n_clipped: int = 0      # diagnostic: how many (scenario, day) exponents hit the clip bound


def estimate_price_demand_elasticity(log_returns: np.ndarray, volumes: np.ndarray) -> float:
    """
    Proposal 3.4 step 3: "elastisitas harga-permintaan empiris yang
    diestimasi dari data historis". The proposal does not specify the
    estimator - this uses the standard log-log regression:

        ln(volume_t) = a + beta * r_t + eps_t

    beta is the elasticity of demand (volume) with respect to same-day
    log-return. Estimated via OLS (closed form, no external dependency).

    NOTE: this is a first-order, static/linear elasticity. It says
    nothing about lagged effects or asymmetry (does a +5% day pull in as
    much extra demand as a -5% day pushes out?) - both are plausible in
    crypto and neither is tested here. Flag this as a modeling choice,
    not settled fact, if you write this up.
    """
    log_vol = np.log(np.clip(volumes, 1e-8, None))
    x = log_returns
    y = log_vol
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    beta = np.sum((x - x_mean) * (y - y_mean)) / denom
    return float(beta)


def simulate_demand_scenarios(historical_log_returns: np.ndarray,
                               historical_volume_mean: float,
                               elasticity: float,
                               lead_time_days: int,
                               n_scenarios: int,
                               rng: np.random.Generator,
                               exponent_clip: float = config.EXPONENT_CLIP) -> DemandScenarioOutput:
    """
    Runs steps 1-4 of proposal section 3.4 for a single "as-of" day.

    historical_log_returns: 1-D array of log-returns from data available
        up to (and including) the as-of day - this is what makes the
        re-run "walk-forward" rather than using the full future dataset.
    historical_volume_mean: average daily volume over the same window,
        used as the demand baseline that price shocks scale around.
    elasticity: beta from estimate_price_demand_elasticity, computed once
        on the calibration window (re-estimating it daily is possible but
        the proposal does not require it; document whichever you choose).
    exponent_clip: bounds |elasticity * r| before exponentiating (see
        module docstring "BUG FIX" section). Set to np.inf to reproduce
        the original unbounded (buggy) behavior for comparison purposes
        only -- never use np.inf for a reported result.
    """
    n = len(historical_log_returns)
    if n == 0:
        raise ValueError("Need at least one historical log-return to bootstrap from.")
    if historical_volume_mean < 0:
        raise ValueError("historical_volume_mean must be non-negative.")

    # Step 1+2: bootstrap L sequential days with replacement, for each of N scenarios
    draws = rng.choice(historical_log_returns, size=(n_scenarios, lead_time_days), replace=True)

    # Step 3: convert each day's return into a demand multiplier via elasticity,
    # then scale the historical baseline volume by that multiplier.
    # demand_t = volume_mean * exp(clip(elasticity * r_t, -clip, +clip))
    # (log-linear scaling, keeps demand strictly positive regardless of
    # elasticity sign/magnitude; clip bounds it -- see module docstring)
    raw_exponent = elasticity * draws
    clipped_exponent = np.clip(raw_exponent, -exponent_clip, exponent_clip)
    n_clipped = int(np.sum(raw_exponent != clipped_exponent))
    with np.errstate(over="raise"):
        demand_paths = historical_volume_mean * np.exp(clipped_exponent)   # shape (N, L)

    if not np.all(np.isfinite(demand_paths)):
        raise FloatingPointError(
            "Non-finite demand scenario produced even after clipping -- "
            "this should not happen with a finite exponent_clip. Check "
            "historical_volume_mean and elasticity for NaN/inf inputs."
        )

    # Step 4: sigma_d / d_bar from the simulated *daily* demand values
    daily_demand_flat = demand_paths.flatten()
    sigma_d = float(np.std(daily_demand_flat, ddof=1))
    d_bar = float(np.mean(daily_demand_flat))

    # target inventory input: P95 of the L-day CUMULATIVE demand per scenario
    cumulative_demand_per_scenario = demand_paths.sum(axis=1)
    p95_demand = float(np.percentile(cumulative_demand_per_scenario, 95))

    return DemandScenarioOutput(sigma_d=sigma_d, d_bar=d_bar, p95_demand=p95_demand,
                                 scenarios=demand_paths, n_clipped=n_clipped)


def walk_forward_daily_params(log_returns: pd.Series, volume: pd.Series,
                               elasticity: float, lead_time_days: int,
                               n_scenarios: int, seed: int,
                               min_history_days: int = 180,
                               exponent_clip: float = config.EXPONENT_CLIP) -> pd.DataFrame:
    """
    Re-runs the Monte Carlo simulation for every trading day in the index
    of `log_returns`/`volume`, using only data up to (and including) that
    day - i.e. a proper walk-forward / expanding-window design, matching
    "Simulasi dijalankan ulang setiap hari perdagangan" (3.4).

    Returns a DataFrame indexed by date with columns: sigma_d, d_bar,
    p95_demand, n_clipped. n_clipped > 0 on a given day is not an error --
    it's a diagnostic showing how often the clip bound bound was hit; if
    it is large and growing across the backtest, the exponent_clip choice
    is materially affecting results and must be reported, not buried.

    WARNING ON COST: this is O(days * N_scenarios * L). With N=10,000 and
    ~365 backtest days this is fine (seconds). Do not naively scale
    n_scenarios up by 10x without checking runtime first.
    """
    rng = np.random.default_rng(seed)
    dates = log_returns.index
    results = []

    for i, current_date in enumerate(dates):
        if i + 1 < min_history_days:
            continue  # not enough history yet to bootstrap sensibly
        hist_returns = log_returns.iloc[: i + 1].values
        hist_vol_mean = volume.iloc[: i + 1].mean()

        out = simulate_demand_scenarios(
            historical_log_returns=hist_returns,
            historical_volume_mean=hist_vol_mean,
            elasticity=elasticity,
            lead_time_days=lead_time_days,
            n_scenarios=n_scenarios,
            rng=rng,
            exponent_clip=exponent_clip,
        )
        results.append({
            "date": current_date,
            "sigma_d": out.sigma_d,
            "d_bar": out.d_bar,
            "p95_demand": out.p95_demand,
            "n_clipped": out.n_clipped,
        })

    return pd.DataFrame(results).set_index("date")
