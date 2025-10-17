# %%
#!/usr/bin/env python3

"""
metrics_viz_suite.py

Create publication-ready plots from the long-form overhead CSV:
  metric, source, api, iters, ok_count, min_us, avg_us, max_us, stddev_us, notes

What you get (choose individually or --all):
  - Butterfly split violin (AMDSMI vs PAPI, aggregated)
    * Shows mean & median for each side
    * NEW: Shows Standard Deviation (from stddev_us) below the mean for each side
  - Slope (dumbbell) plot per metric
  - Scatter PAPI vs AMDSMI (with y=x)
  - Histogram of (PAPI - AMDSMI)
  - Bland–Altman plot (difference vs mean)

Example:
  python metrics_viz_suite.py \
    --csv overhead_synced.csv \
    --value-col avg_us \
    --out-prefix overhead \
    --all
"""

from __future__ import annotations
import argparse, sys, math, re
from typing import Tuple, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None
    print("This script needs pandas. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)


# ------------------ utils ------------------
def load_metrics(csv_path: str) -> "pd.DataFrame":
    # Robust delimiter detection (comma/tab/space)
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(csv_path, sep="\t")
        except Exception:
            df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def pairs_by_metric(df: "pd.DataFrame", value_col: str,
                    left_src: str = "AMDSMI",
                    right_src: str = "PAPI") -> List[Tuple[str, float, float]]:
    """
    Returns [(metric_name, left_val, right_val), ...] for rows where both sources exist.
    """
    g = df.pivot_table(index="metric", columns="source", values=value_col, aggfunc="mean")
    g = g.dropna(subset=[left_src, right_src])
    out = [(m, float(g.at[m, left_src]), float(g.at[m, right_src])) for m in g.index]
    return out


def shorten_metric(m: str) -> str:
    """
    Compact the metric label: e.g.
      "amd_smi:::temp_current:device=0:sensor=3" -> "temp_current[s=3]"
      "amd_smi:::gfx_activity:device=0"         -> "gfx_activity"
    """
    m2 = m.replace("amd_smi:::", "")
    m2 = re.sub(r"device=\d+", "", m2)
    m2 = m2.replace(":::", ":").strip(":")
    m2 = m2.replace("  ", " ")
    # compress sensor
    m2 = re.sub(r":sensor=(\d+)", r"[s=\1]", m2)
    m2 = m2.replace(":]", "]").replace("[]", "")
    return m2


# ------------------ tiny KDE (NumPy only) ------------------
def _silverman_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return 1.0
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.349) if std > 0 else (iqr / 1.349 if iqr > 0 else 1.0)
    bw = 0.9 * sigma * (n ** (-1/5))
    return bw if bw > 0 else 1.0

def _kde_1d(samples: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    s = np.asarray(samples, dtype=float).ravel()
    h = _silverman_bandwidth(s)
    u = (y_grid[:, None] - s[None, :]) / h
    K = np.exp(-0.5 * (u**2)) / np.sqrt(2*np.pi)
    dens = K.mean(axis=1) / h
    return dens


# ------------------ plots ------------------
def plot_butterfly_improved(left: np.ndarray, right: np.ndarray,
                            title: str, left_label: str, right_label: str,
                            ylabel: str = "Time in µs", width: float = 0.9,
                            grid_points: int = 256, pad_frac: float = 0.08,
                            label_offset_frac: float = 0.06,
                            left_stddev: Optional[float] = None,
                            right_stddev: Optional[float] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Split (butterfly) violin with:
      - clearer center line
      - 50th percentile (median) bands
      - top-corner mean + (optional) stddev + median labels
    """
    left = np.asarray(left, float).ravel()
    right = np.asarray(right, float).ravel()

    y_min = min(left.min(), right.min())
    y_max = max(left.max(), right.max())
    y_pad = (y_max - y_min) * pad_frac if y_max > y_min else 1.0
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, grid_points)

    dl = _kde_1d(left, y_grid)
    dr = _kde_1d(right, y_grid)
    scale = max(dl.max(), dr.max()) or 1.0
    xl = -(dl/scale)*width
    xr = +(dr/scale)*width

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    # half violins
    left_poly  = ax.fill_betweenx(y_grid, 0, xl, alpha=0.65, lw=0)
    right_poly = ax.fill_betweenx(y_grid, 0, xr, alpha=0.65, lw=0)
    # centerline
    ax.axvline(0, lw=2)

    # medians (50th) as black lines
    ql = np.percentile(left,  [50])
    qr = np.percentile(right, [50])
    med_l = float(ql[0])
    med_r = float(qr[0])
    lc = left_poly.get_facecolor()[0]
    rc = right_poly.get_facecolor()[0]
    for q in ql: ax.plot([0, -width*0.95], [q, q], lw=1.8, color="black")
    for q in qr: ax.plot([0,  width*0.95], [q, q], lw=1.8, color="black")

    # means (colored lines)
    mean_l = float(np.mean(left))
    mean_r = float(np.mean(right))
    ax.plot([0, -width*0.95], [mean_l, mean_l], lw=2.5, color=lc)
    ax.plot([0,  width*0.95], [mean_r, mean_r], lw=2.5, color=rc)

    # top-corner labels positioned slightly below the top
    y0, y1 = y_grid[0], y_grid[-1]
    label_y = y1 - label_offset_frac * (y1 - y0)
    vendor = left_label.split()[0]   # "Vendor call Overhead" -> "Vendor"
    total  = right_label.split()[0]  # "PAPI_read() Overhead" -> "PAPI_read()"

    # Build label text: mean, stddev (if provided), median
    left_lines = [f"{vendor} mean: {mean_l:.2f}µs"]
    if (left_stddev is not None) and np.isfinite(left_stddev):
        left_lines.append(f"{vendor} stddev: {left_stddev:.2f}µs")
    left_lines.append(f"{vendor} median: {med_l:.2f}µs")

    right_lines = [f"{total} mean: {mean_r:.2f}µs"]
    if (right_stddev is not None) and np.isfinite(right_stddev):
        right_lines.append(f"{total} stddev: {right_stddev:.2f}µs")
    right_lines.append(f"{total} median: {med_r:.2f}µs")

    ax.text(-width*0.98, label_y, "\n".join(left_lines),
            ha="left", va="top", fontsize=11, color=lc)
    ax.text( width*0.98, label_y, "\n".join(right_lines),
            ha="right", va="top", fontsize=11, color=rc)

    ax.set_ylim(y_grid[0], y_grid[-1])
    ax.set_xlim(-width*1.15, width*1.15)
    ax.set_xticks([-width*0.6,  width*0.6], [left_label, right_label])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, fontsize=14)
    ax.grid(True, axis="y", alpha=0.25)
    return fig, ax



def plot_slope_per_metric(pairs: List[Tuple[str,float,float]],
                          title: str, value_label: str = "Time in µs") -> Tuple[plt.Figure, plt.Axes]:
    """
    Dumbbell/slope: one line per metric connecting AMDSMI (left dot) to PAPI (right dot).
    Sorted by (PAPI - AMDSMI).
    """
    # pairs = [(metric, amdsmi, papi), ...]
    if not pairs:
        raise ValueError("No pairs to plot.")
    pairs_sorted = sorted(pairs, key=lambda t: (t[2]-t[1]))
    labels = [shorten_metric(m) for m, _, _ in pairs_sorted]
    left   = [t[1] for t in pairs_sorted]
    right  = [t[2] for t in pairs_sorted]
    idx = np.arange(len(pairs_sorted))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35*len(idx))), constrained_layout=True)
    for i in idx:
        ax.plot([left[i], right[i]], [i, i], lw=2, alpha=0.6)  # slope line
    ax.scatter(left,  idx, s=30, label="AMDSMI")
    ax.scatter(right, idx, s=30, label="PAPI")

    ax.set_yticks(idx, labels)
    ax.set_xlabel(value_label)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    return fig, ax


def plot_scatter_parity(pairs: List[Tuple[str,float,float]],
                        title: str, value_label: str = "Time in µs") -> Tuple[plt.Figure, plt.Axes]:
    """Scatter of PAPI vs AMDSMI with y=x parity line."""
    if not pairs: raise ValueError("No pairs to plot.")
    left  = np.array([t[1] for t in pairs])
    right = np.array([t[2] for t in pairs])

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(left, right, s=30)
    lo = min(left.min(), right.min())
    hi = max(left.max(), right.max())
    ax.plot([lo, hi], [lo, hi], lw=2)  # y=x
    ax.set_xlabel("AMDSMI (µs)")
    ax.set_ylabel("PAPI (µs)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_diff_hist(pairs: List[Tuple[str,float,float]],
                   title: str, bins: int = 15) -> Tuple[plt.Figure, plt.Axes]:
    """Histogram of (PAPI - AMDSMI)."""
    diffs = np.array([t[2]-t[1] for t in pairs])
    mu = diffs.mean()
    sd = diffs.std(ddof=1) if len(diffs) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.hist(diffs, bins=bins, alpha=0.7, edgecolor="white")
    ax.axvline(0, lw=2)
    ax.axvline(mu, lw=2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("PAPI − AMDSMI (µs)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.99, 0.95, f"mean={mu:.2f}µs, σ={sd:.2f}µs",
        ha="right", va="top", transform=ax.transAxes)
    return fig, ax


def plot_bland_altman(pairs: List[Tuple[str,float,float]],
                      title: str) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bland–Altman: diff vs mean with bias and ±1.96σ.
    """
    left  = np.array([t[1] for t in pairs])
    right = np.array([t[2] for t in pairs])
    mean  = (left + right) / 2.0
    diff  = (right - left)

    mu = diff.mean()
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    loA = mu - 1.96 * sd
    hiA = mu + 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.scatter(mean, diff, s=30, alpha=0.8)
    ax.axhline(mu,  lw=2, linestyle="--")
    ax.axhline(loA, lw=1.5, linestyle=":")
    ax.axhline(hiA, lw=1.5, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Mean of AMDSMI & PAPI (µs)")
    ax.set_ylabel("PAPI − AMDSMI (µs)")
    ax.grid(True, alpha=0.3)
    ax.text(0.01, 0.98, f"bias={mu:.2f}µs  (±1.96σ → {loA:.2f}, {hiA:.2f})",
            ha="left", va="top", transform=ax.transAxes)
    return fig, ax


# ------------------ CLI ------------------
def main():
    p = argparse.ArgumentParser(description="Visualize AMDSMI vs PAPI overheads.")
    p.add_argument("--csv", required=True, help="Long-form CSV/TSV with columns incl. 'metric','source','avg_us'.")
    p.add_argument("--value-col", default="avg_us", choices=["min_us","avg_us","max_us"],
                   help="Which column to visualize (default: avg_us).")
    p.add_argument("--stddev-col", default="stddev_us",
                   help="Per-row stddev column (µs) to aggregate and display under each mean in the butterfly plot.")
    p.add_argument("--metric-regex", default=None,
                   help="Optional regex to subset metrics (e.g., 'temp_current').")
    p.add_argument("--left-source", default="AMDSMI", help="Left side source (default: AMDSMI).")
    p.add_argument("--right-source", default="PAPI",   help="Right side source (default: PAPI).")
    p.add_argument("--title", default="Overhead of PAPI_read() in sampling mode")
    p.add_argument("--left-label", default="Vendor call Overhead")
    p.add_argument("--right-label", default="PAPI_read() Overhead")
    p.add_argument("--out-prefix", default="overhead", help="Filename prefix for outputs.")
    p.add_argument("--butterfly", action="store_true")
    p.add_argument("--slope", action="store_true")
    p.add_argument("--scatter", action="store_true")
    p.add_argument("--hist", action="store_true")
    p.add_argument("--bland", action="store_true")
    p.add_argument("--all", action="store_true",
                   help="Produce all plots.")
    args = p.parse_args()

    df = load_metrics(args.csv)
    if args.metric_regex:
        rx = re.compile(args.metric_regex)
        df = df[df["metric"].astype(str).str.match(rx)]
        if df.empty:
            sys.exit("No rows match --metric-regex.")

    # Build paired data per metric for pairwise charts
    pairs = pairs_by_metric(df, args.value_col, args.left_source, args.right_source)

    # Aggregated vectors for butterfly (all samples across metrics per source)
    left_vals  = pd.to_numeric(df[df["source"] == args.left_source][args.value_col], errors="coerce").dropna().to_numpy()
    right_vals = pd.to_numeric(df[df["source"] == args.right_source][args.value_col], errors="coerce").dropna().to_numpy()
    if left_vals.size == 0 or right_vals.size == 0:
        sys.exit(f"No data for sources {args.left_source} / {args.right_source} using {args.value_col}")

    # Aggregate stddev per source (mean of per-row stddevs). If missing, omit.
    left_stddev = right_stddev = None
    if args.stddev_col in df.columns:
        left_std_vals  = pd.to_numeric(
            df.loc[df["source"] == args.left_source,  args.stddev_col],
            errors="coerce"
        ).dropna()
        right_std_vals = pd.to_numeric(
            df.loc[df["source"] == args.right_source, args.stddev_col],
            errors="coerce"
        ).dropna()
        if not left_std_vals.empty:
            left_stddev = float(left_std_vals.mean())
        if not right_std_vals.empty:
            right_stddev = float(right_std_vals.mean())

    do_all = args.all or not any([args.butterfly, args.slope, args.scatter, args.hist, args.bland])

    if args.butterfly or do_all:
        fig, ax = plot_butterfly_improved(
            left_vals, right_vals,
            title=args.title,
            left_label=args.left_label,
            right_label=args.right_label,
            ylabel="Time in µs",
            left_stddev=left_stddev,
            right_stddev=right_stddev,
        )
        fn = f"{args.out_prefix}_butterfly.png"
        fig.savefig(fn, dpi=200)
        print(f"saved {fn}")

    if (args.slope or do_all) and pairs:
        fig, ax = plot_slope_per_metric(
            pairs, title=f"{args.value_col} per metric (AMDSMI vs PAPI)"
        )
        fn = f"{args.out_prefix}_slope.png"
        fig.savefig(fn, dpi=200)
        print(f"saved {fn}")

    if (args.scatter or do_all) and pairs:
        fig, ax = plot_scatter_parity(
            pairs, title=f"PAPI vs AMDSMI ({args.value_col})"
        )
        fn = f"{args.out_prefix}_scatter.png"
        fig.savefig(fn, dpi=200)
        print(f"saved {fn}")

    if (args.hist or do_all) and pairs:
        fig, ax = plot_diff_hist(
            pairs, title=f"Distribution of (PAPI − AMDSMI) for {args.value_col}"
        )
        fn = f"{args.out_prefix}_diff_hist.png"
        fig.savefig(fn, dpi=200)
        print(f"saved {fn}")

    if (args.bland or do_all) and pairs:
        fig, ax = plot_bland_altman(
            pairs, title=f"Bland–Altman: PAPI vs AMDSMI ({args.value_col})"
        )
        fn = f"{args.out_prefix}_bland_altman.png"
        fig.savefig(fn, dpi=200)
        print(f"saved {fn}")


if __name__ == "__main__":
    main()
