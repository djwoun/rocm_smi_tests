#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AMD SMI GEMM monitor output (amdsmi_gemm_*.csv)."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to amdsmi_gemm_*.csv produced by --test mode.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device index recorded in the CSV (default: 0).",
    )
    parser.add_argument(
        "--sensor",
        type=int,
        default=1,
        help="Temperature sensor id to plot (default: 1 == Edge).",
    )
    parser.add_argument(
        "--no-power",
        dest="plot_power",
        action="store_false",
        help="Skip plotting power_current if the column is present.",
    )
    parser.add_argument(
        "--proc",
        type=int,
        default=0,
        help="Process index used in process_* columns (default: 0).",
    )
    parser.add_argument(
        "--max-cu",
        type=int,
        default=None,
        help="If provided, scale CU occupancy by this compute-unit count "
        "and plot it alongside activity percentages.",
    )
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(
            f"Required column '{column}' not found in {df.columns.tolist()}"
        )
    return df[column]


def main() -> int:
    args = parse_args()
    if not args.csv.is_file():
        print(f"error: {args.csv} does not exist", file=sys.stderr)
        return 1
    if args.proc < 0:
        print("error: --proc must be non-negative", file=sys.stderr)
        return 1
    if args.max_cu is not None and args.max_cu <= 0:
        print("error: --max-cu must be positive when specified", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    if "timestamp" not in df.columns:
        print("error: CSV is missing 'timestamp' column", file=sys.stderr)
        return 1
    df["time_s"] = df["timestamp"] - df["timestamp"].iloc[0]

    device = args.device
    sensor = args.sensor

    temp_col = f"amd_smi:::temp_current_sensor={sensor}:device={device}"
    junction_col = f"amd_smi:::temp_current_sensor=2:device={device}"
    memory_col = f"amd_smi:::temp_current_sensor=7:device={device}"
    gfx_col = f"amd_smi:::gfx_activity:device={device}"
    umc_col = f"amd_smi:::umc_activity:device={device}"
    power_col = f"amd_smi:::power_average:device={device}"
    cu_col = (
        f"amd_smi:::process_u_occupancy_proc={args.proc}:device={device}"
    )

    try:
        temp_series = resolve_column(df, temp_col).astype(float)
        gfx_series = resolve_column(df, gfx_col).astype(float)
        umc_series = resolve_column(df, umc_col).astype(float)
        junction_series = resolve_column(df, junction_col).astype(float)
        memory_series = resolve_column(df, memory_col).astype(float)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    power_series = None
    if args.plot_power and power_col in df.columns:
        power_series = df[power_col].astype(float)

    cu_series = None
    cu_percent_series = None
    if cu_col in df.columns:
        cu_series = df[cu_col].astype(float)
        if args.max_cu is not None:
            cu_percent_series = cu_series / float(args.max_cu) * 100.0

    sensor_names = {
        0: "Unknown",
        1: "Edge",
        2: "Junction",
        3: "Memory",
        4: "VR SOC",
        5: "HBM VR",
        6: "HBM",
        7: "PXT",
    }
    sensor_name = sensor_names.get(sensor, f"Sensor {sensor}")

    fig, ax_temp = plt.subplots(figsize=(20, 6))
    (line_temp,) = ax_temp.plot(
        df["time_s"],
        temp_series,
        color="tab:red",
        linewidth=2.0,
        label=f"{sensor_name} Temperature",
    )
    (line_junc,) = ax_temp.plot(
        df["time_s"],
        junction_series,
        color="tab:purple",
        linewidth=1.5,
        linestyle="-",
        label="Junction Temperature",
    )
    (line_mem,) = ax_temp.plot(
        df["time_s"],
        memory_series,
        color="tab:brown",
        linewidth=1.5,
        linestyle="-",
        label="PXT Temperature",
    )

    ax_temp.set_xlabel("Time (s)")
    ax_temp.set_ylabel("Temperature (°C)", color="tab:red")
    ax_temp.tick_params(axis="y", labelcolor="tab:red")
    ax_temp.set_ylim(0, 105)
    ax_temp.set_yticks(range(0, 106, 5))
    ax_temp.yaxis.grid(True, linestyle="--", linewidth=0.5)

    ax_activity = ax_temp.twinx()
    (line_gfx,) = ax_activity.plot(
        df["time_s"],
        gfx_series,
        color="blue",
        linewidth=1.8,
        linestyle="--",
        label="GFX Activity (%)",
    )
    (line_umc,) = ax_activity.plot(
        df["time_s"],
        umc_series,
        color="#d95f02",
        linewidth=1.8,
        linestyle="--",
        label="UMC Activity (%)",
    )
    activity_limits = [gfx_series.max(), umc_series.max()]
    if cu_percent_series is not None:
        activity_limits.append(cu_percent_series.max())
    activity_max = max(activity_limits) if activity_limits else 0.0
    activity_upper = max(105, math.ceil(activity_max / 5.0) * 5) if activity_max else 105
    ax_activity.set_ylabel(
        "Activity (%)"
        if cu_percent_series is None
        else "Activity / CU Occupancy (%)",
        color="blue",
    )
    ax_activity.tick_params(axis="y", labelcolor="blue")
    ax_activity.set_ylim(0, activity_upper)
    ax_activity.set_yticks(range(0, int(activity_upper) + 1, 5))

    lines = [line_temp, line_junc, line_mem, line_gfx, line_umc]
    labels = [line.get_label() for line in lines]

    extra_axis_offset = 1.0

    def create_offset_axis(base_ax, offset):
        ax_off = base_ax.twinx()
        ax_off.spines["right"].set_position(("axes", offset))
        ax_off.set_frame_on(True)
        ax_off.patch.set_visible(False)
        for spine in ax_off.spines.values():
            spine.set_visible(False)
        ax_off.spines["right"].set_visible(True)
        return ax_off

    if power_series is not None:
        extra_axis_offset += 0.1
        ax_power = create_offset_axis(ax_temp, extra_axis_offset)
        (line_power,) = ax_power.plot(
            df["time_s"],
            power_series,
            color="#02320C",
            linewidth=1.6,
            linestyle="--",
            label="Power (W)",
        )
        ax_power.set_ylabel("Power (W)", color="#02320C")
        ax_power.tick_params(axis="y", labelcolor="#02320C")
        ymin = max(0.0, power_series.min() // 50 * 50)
        ymax = (power_series.max() // 50 + 1) * 50
        if ymin == ymax:
            ymax = ymin + 50
        ax_power.set_ylim(ymin, ymax)
        ax_power.set_yticks(range(int(ymin), int(ymax) + 1, 50))
        lines.append(line_power)
        labels.append(line_power.get_label())

    if cu_series is not None:
        cu_color = "#ff69b4"
        if cu_percent_series is not None:
            (line_cu,) = ax_activity.plot(
                df["time_s"],
                cu_percent_series,
                color=cu_color,
                linewidth=1.8,
                linestyle=":",
                label=f"CU Occupancy (% of {args.max_cu})",
            )
            lines.append(line_cu)
            labels.append(line_cu.get_label())
        else:
            extra_axis_offset += 0.1
            ax_cu = create_offset_axis(ax_temp, extra_axis_offset)
            (line_cu,) = ax_cu.plot(
                df["time_s"],
                cu_series,
                color=cu_color,
                linewidth=1.8,
                linestyle=":",
                label="CU Occupancy (CUs)",
            )
            cu_upper = cu_series.max() if not cu_series.empty else 0.0
            cu_upper = max(10.0, math.ceil(cu_upper / 10.0) * 10)
            ax_cu.set_ylabel("CU Occupancy (CUs)", color=cu_color)
            ax_cu.tick_params(axis="y", labelcolor=cu_color)
            ax_cu.set_ylim(0, cu_upper)
            ax_cu.set_yticks(range(0, int(cu_upper) + 1, 10))
            lines.append(line_cu)
            labels.append(line_cu.get_label())

    ax_temp.set_title("AMD-SMI GEMM Monitor - MI250X")
    t_min = df["time_s"].min()
    t_max = df["time_s"].max()
    ax_temp.set_xlim(t_min, t_max)
    xtick_min = int(t_min)
    xtick_max = int(t_max)
    ax_temp.set_xticks(range(xtick_min, xtick_max + 1, 5))
    ax_temp.legend(
        lines,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.25, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )
    ax_temp.xaxis.grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
