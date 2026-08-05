"""
Unit tests for the elasticity-scaling bug fix in scripts/monte_carlo.py.

Run before touching real data (per Fase 0 step 1):
    cd Luaran/tests
    python -m pytest test_monte_carlo_elasticity_fix.py -v
  or, if pytest isn't installed:
    python test_monte_carlo_elasticity_fix.py

These tests reproduce the exact overflow scenario documented in
Draft_Artikel_Preliminer_Monte_Carlo_Inventori_Kripto_rev.docx (Bagian 5)
that made ETH's preliminary stockout metrics non-physical, and confirm
the clipped version stays finite and bounded under the same conditions.
"""

import sys
from pathlib import Path

# PATH FIX (post folder-reorg): monte_carlo.py now lives in
# Luaran/scripts/00_modul_inti/, not directly under scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "00_modul_inti"))

import numpy as np
import monte_carlo


def test_original_unclipped_formula_overflows():
    """
    Confirms the ORIGINAL bug is real and reproducible. The OLS elasticity
    estimator (beta = cov(r, log_volume) / var(r)) is unstable when the
    calibration window's return variance is small: var(r) in the
    denominator can shrink enough during a calm sub-period that beta
    spikes into the hundreds or low thousands. That inflated beta, applied
    later to a crisis-magnitude return once the walk-forward window
    absorbs one (FTX collapse, ~-55% single day), sends
    exp(elasticity * r) into overflow under the unclipped formula. This is
    not a strawman -- it's the literal computation from the original
    notebook's monte_carlo.py before the fix, with an elasticity magnitude
    that a low-variance calibration sub-window can genuinely produce.
    """
    elasticity = 1500.0        # degenerate high-end OLS estimate (small var(r) denominator)
    extreme_return = 0.55      # a crisis-magnitude single-day log-return (positive shock case)
    raw_exponent = elasticity * extreme_return
    with np.errstate(over="ignore"):
        result = np.exp(raw_exponent)
    assert not np.isfinite(result), (
        f"expected overflow to reproduce the original bug, got finite value {result}. "
        f"If this assertion fails, the reproduction case needs a more extreme input."
    )


def test_clipped_formula_stays_finite_under_same_extreme_input():
    """The fixed simulate_demand_scenarios must never overflow, no matter
    how extreme the historical return or elasticity is."""
    rng = np.random.default_rng(0)
    # Inject the extreme FTX-magnitude return directly into the historical
    # pool so it can be drawn by the bootstrap.
    historical_returns = np.array([-0.55, -0.10, 0.02, 0.15, -0.03, 0.55, 0.01])
    out = monte_carlo.simulate_demand_scenarios(
        historical_log_returns=historical_returns,
        historical_volume_mean=1_000_000.0,
        elasticity=1500.0,
        lead_time_days=1,
        n_scenarios=10_000,
        rng=rng,
        exponent_clip=10.0,
    )
    assert np.all(np.isfinite(out.scenarios)), "demand scenarios contain non-finite values"
    assert out.sigma_d > 0
    assert out.d_bar > 0
    # bounded by construction: max possible multiplier is exp(10) ~= 22026x
    assert out.scenarios.max() <= 1_000_000.0 * np.exp(10.0) * 1.0001


def test_clip_diagnostic_counts_correctly():
    """n_clipped should be > 0 when the clip bound is actually hit, and
    exactly 0 when elasticity/returns are mild enough never to hit it."""
    rng = np.random.default_rng(1)
    historical_returns = np.array([-0.55, 0.55, 0.01, -0.02])

    out_hit = monte_carlo.simulate_demand_scenarios(
        historical_log_returns=historical_returns,
        historical_volume_mean=1000.0,
        elasticity=1500.0,
        lead_time_days=1,
        n_scenarios=1000,
        rng=rng,
        exponent_clip=10.0,
    )
    assert out_hit.n_clipped > 0, "expected the clip to be hit with elasticity=1500, |r|=0.55"

    rng2 = np.random.default_rng(2)
    mild_returns = np.array([0.001, -0.002, 0.0005, -0.0015])
    out_mild = monte_carlo.simulate_demand_scenarios(
        historical_log_returns=mild_returns,
        historical_volume_mean=1000.0,
        elasticity=0.5,
        lead_time_days=1,
        n_scenarios=1000,
        rng=rng2,
        exponent_clip=10.0,
    )
    assert out_mild.n_clipped == 0, "did not expect the clip to be hit with mild inputs"


def test_zero_elasticity_gives_flat_demand():
    """Sanity check: elasticity=0 means price moves don't scale demand at
    all -- every scenario should equal the historical mean exactly."""
    rng = np.random.default_rng(3)
    out = monte_carlo.simulate_demand_scenarios(
        historical_log_returns=np.array([0.1, -0.1, 0.05]),
        historical_volume_mean=500.0,
        elasticity=0.0,
        lead_time_days=1,
        n_scenarios=100,
        rng=rng,
        exponent_clip=10.0,
    )
    assert np.allclose(out.scenarios, 500.0)
    assert out.sigma_d == 0.0
    assert out.d_bar == 500.0


def test_rejects_negative_volume_mean():
    rng = np.random.default_rng(4)
    try:
        monte_carlo.simulate_demand_scenarios(
            historical_log_returns=np.array([0.01, -0.01]),
            historical_volume_mean=-5.0,
            elasticity=1.0,
            lead_time_days=1,
            n_scenarios=10,
            rng=rng,
        )
        assert False, "expected ValueError for negative historical_volume_mean"
    except ValueError:
        pass


def test_walk_forward_never_produces_nan_or_inf_across_expanding_window():
    """
    Simulates the exact failure mode from the preliminary ETH run: a
    long expanding walk-forward window that eventually absorbs an
    extreme historical return. Confirms sigma_d/d_bar/p95_demand stay
    finite for every single day of the walk-forward, not just on average.
    """
    import pandas as pd
    rng_seed = 42
    n_days = 400
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    returns = np.random.default_rng(5).normal(0, 0.03, n_days)
    returns[250] = -0.55  # inject one FTX-magnitude day partway through
    volume = np.abs(np.random.default_rng(6).normal(1_000_000, 200_000, n_days))
    volume = np.clip(volume, 1000, None)

    log_returns = pd.Series(returns, index=dates)
    vol_series = pd.Series(volume, index=dates)

    result = monte_carlo.walk_forward_daily_params(
        log_returns=log_returns, volume=vol_series,
        elasticity=1500.0, lead_time_days=1, n_scenarios=2000,
        seed=rng_seed, min_history_days=100, exponent_clip=10.0,
    )
    assert result[["sigma_d", "d_bar", "p95_demand"]].apply(np.isfinite).all().all(), (
        "found non-finite sigma_d/d_bar/p95_demand somewhere in the walk-forward output"
    )
    assert (result["n_clipped"] >= 0).all()
    # after day 250 is absorbed into the expanding window, later days must
    # show the clip being exercised at least once (that's the whole point
    # of this test: extreme historical events entering the bootstrap pool)
    after_crisis = result.loc[result.index > dates[250]]
    assert after_crisis["n_clipped"].sum() > 0, (
        "expected the clip to be exercised at least once after the injected "
        "extreme return enters the expanding window -- if this fails, the "
        "reproduction scenario isn't actually stressing the fix"
    )


if __name__ == "__main__":
    tests = [
        test_original_unclipped_formula_overflows,
        test_clipped_formula_stays_finite_under_same_extreme_input,
        test_clip_diagnostic_counts_correctly,
        test_zero_elasticity_gives_flat_demand,
        test_rejects_negative_volume_mean,
        test_walk_forward_never_produces_nan_or_inf_across_expanding_window,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
