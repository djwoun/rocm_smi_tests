
#!/usr/bin/env python3
"""
Compute geometric-mean call-time ratio (PAPI / AMDSMI) and its 98% CI
using the pooled log-ratio method described in the paper.
Accepts either:
  - aggregated CSV with columns: metric, source, avg_us  (e.g., overhead_iter500.csv), or
  - per-iteration CSV with columns: metric, source, iteration, us (e.g., overhead_iter500_samples.csv).

Usage:
  python call_time_equivalence.py /path/to/overhead_iter500.csv
  python call_time_equivalence.py /path/to/overhead_iter500_samples.csv
"""
import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


def _z_quantile(p: float) -> float:
    """Return z such that P(Z <= z) = p for Z~N(0,1)."""
    return NormalDist().inv_cdf(p)


def _t_quantile_approx(p: float, df: int) -> float:
    """
    Cornish–Fisher style approximation for the Student-t quantile.
    Accurate to ~3-4 decimals for df>=10. Falls back to normal as df->inf.
    """
    if df <= 0:
        raise ValueError("df must be positive")
    z = _z_quantile(p)
    # Cornish-Fisher expansion terms for t-quantiles
    z3 = z**3
    z5 = z**5
    term1 = (z3 + z) / (4 * df)
    term2 = (5 * z5 + 16 * z3 + 3 * z) / (96 * df**2)
    return z + term1 + term2


def load_times(path: Path) -> pd.DataFrame:
    """
    Load input CSV. Return a frame with columns [metric, source, mean_us].
    - If 'avg_us' exists, use it directly.
    - Else expect per-iteration 'us' and compute mean per metric+source.
    """
    df = pd.read_csv(path)
    # Normalize source labels if needed
    # Expect 'PAPI' for the counter interface and 'AMDSMI' for direct library.
    # Attempt light normalization.
    df['source'] = df['source'].replace({'CI': 'PAPI', 'Direct': 'AMDSMI'})
    if 'avg_us' in df.columns:
        # aggregated file
        out = df[['metric', 'source', 'avg_us']].rename(columns={'avg_us': 'mean_us'})
    elif {'metric', 'source', 'iteration', 'us'}.issubset(df.columns):
        # per-iteration file; aggregate properly by metric+source
        out = (
            df.groupby(['metric', 'source'], as_index=False)['us']
              .mean()
              .rename(columns={'us': 'mean_us'})
        )
    else:
        raise ValueError(
            "Unrecognized CSV schema. Need columns (metric, source, avg_us) "
            "or (metric, source, iteration, us)."
        )
    return out


def compute_summary(per_metric_means: pd.DataFrame, alpha: float = 0.02, delta: float = 0.02):
    """
    Given per-metric means with columns [metric, source, mean_us], compute:
      - geometric-mean ratio (PAPI/AMDSMI)
      - 100*(1-alpha)% CI using t critical value with df=K-1
      - equivalence decision vs ±delta on the ratio scale
      - cross-metric means in microseconds
    Returns a dict.
    """
    pivot = per_metric_means.pivot(index='metric', columns='source', values='mean_us')
    for needed in ['PAPI', 'AMDSMI']:
        if needed not in pivot.columns:
            raise ValueError(f"Missing required source '{needed}'. Found sources: {list(pivot.columns)}")

    # Per-metric ratio and log-ratio
    ratios = pivot['PAPI'] / pivot['AMDSMI']
    r = np.log(ratios.values)

    K = len(r)
    if K < 2:
        raise ValueError("Need at least two metrics to compute a CI on the pooled log-ratio.")

    rbar = float(np.mean(r))
    sr = float(np.std(r, ddof=1))

    # t critical for two-sided (1-alpha) CI on the mean with df=K-1
    df_t = K - 1
    p = 1 - alpha / 2.0  # e.g., 0.99 for a 98% CI
    tcrit = _t_quantile_approx(p, df_t)

    half_width = tcrit * sr / math.sqrt(K)

    # Back-transform
    ratio_gmean = math.exp(rbar)
    ci_low = math.exp(rbar - half_width)
    ci_high = math.exp(rbar + half_width)

    # Equivalence bounds on log scale
    log_lo = math.log(1 - delta)
    log_hi = math.log(1 + delta)
    ci_within_equiv = (rbar - half_width >= log_lo) and (rbar + half_width <= log_hi)

    # Cross-metric absolute means for reference (means of per-metric means)
    mean_smi = float(pivot['AMDSMI'].mean())
    mean_papi = float(pivot['PAPI'].mean())

    return {
        'K': K,
        'tcrit': tcrit,
        'rbar': rbar,
        'sr': sr,
        'ratio_gmean': ratio_gmean,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'equivalence_margin': delta,
        'equivalent': ci_within_equiv,
        'cross_metric_mean_AMDSMI_us': mean_smi,
        'cross_metric_mean_PAPI_us': mean_papi,
        'diff_us': mean_papi - mean_smi,
        'diff_pct': (mean_papi - mean_smi) / mean_smi * 100.0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compute pooled geometric-mean call-time ratio (PAPI/AMDSMI) with 98% CI."
    )
    ap.add_argument("csv_path", type=Path, help="Path to overhead_iter500.csv or overhead_iter500_samples.csv")
    ap.add_argument("--alpha", type=float, default=0.02, help="Significance level for CI. Default 0.02 for 98%% CI.")
    ap.add_argument("--delta", type=float, default=0.02, help="Equivalence margin on ratio scale. Default 0.02 (±2%%).")
    args = ap.parse_args()

    per_metric_means = load_times(args.csv_path)
    res = compute_summary(per_metric_means, alpha=args.alpha, delta=args.delta)

    # Print concise summary
    print(f"K={res['K']} metrics; t_crit={res['tcrit']:.4f}; alpha={args.alpha}")
    print(f"Geometric-mean ratio (PAPI/AMDSMI): {res['ratio_gmean']:.6f}")
    print(f"98% CI: [{res['ci_low']:.6f}, {res['ci_high']:.6f}]")
    print(f"Equivalence margin: ±{res['equivalence_margin']*100:.1f}% "
          f"-> {'EQUIVALENT' if res['equivalent'] else 'NOT equivalent'}")
    print(f"Cross-metric mean AMDSMI: {res['cross_metric_mean_AMDSMI_us']:.3f} µs")
    print(f"Cross-metric mean PAPI:   {res['cross_metric_mean_PAPI_us']:.3f} µs")
    print(f"Absolute diff: {res['diff_us']:.3f} µs  ({res['diff_pct']:.2f}%)")


if __name__ == "__main__":
    main()
