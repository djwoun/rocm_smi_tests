# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:56:16 2025

@author: djwou
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Adjust the file name if needed.
csv_file = 'data/odyssey_gemm_7682_65536_1024_961_20RUNS.csv'

# Read the CSV file.
df = pd.read_csv(csv_file)

# Convert timestamps to relative time (seconds from the start).
df['time_s'] = df['timestamp'] - df['timestamp'].iloc[0]

# Define your device and sensor variables.
device_id = 1
sensor_id = 1

temp = ["Edge", "Junction", "Memory"]
temperature_column = f'rocm_smi:::temp_current:device={device_id}:sensor={sensor_id}'

# If your CSV column is literally called "power", just reference it directly.
power_column = 'power'

# Create the figure and the left y-axis (for Temperature).
# Adjust the figure size to suit your needs.
fig, ax1 = plt.subplots(figsize=(90, 6))


# Plot temperature on the left axis (same as before, in °C).
l1, = ax1.plot(
    df['time_s'],
    df[temperature_column] / 1000,
    color='tab:red',
    label=temp[sensor_id] + ' Temperature'
)

ax1.set_xlabel('Time (s)')  # Relative seconds
ax1.set_ylabel(temp[sensor_id] + ' Temperature (°C)', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Add horizontal grid lines for readability on ax1.
ax1.yaxis.grid(True, linestyle='--', linewidth=0.5)

# Improve the x-axis tick placement (you can adjust to your data range).
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces integer ticks if it helps
ax1.xaxis.grid(True, linestyle='--', linewidth=0.5)

# If you know your temperature range, you can set ylim accordingly.
ax1.set_ylim(0, 100)
ax1.set_yticks(range(0, 101, 5))
ax1.set_xticks(range(0, 3600, 10))
# Create the right y-axis for Power.
ax2 = ax1.twinx()
l2, = ax2.plot(
    df['time_s'],
    df[power_column],
    color='tab:blue',
    label='Power (W)'
)

ax2.set_ylabel('Power (W)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Adjust y-limits for power if you'd like—here, we assume 0-500 W.
ax2.set_ylim(0, 500)
ax2.set_yticks(range(0, 501, 20))
# Create a combined legend for all lines.
lines = [l1, l2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title('Time vs. ' + temp[sensor_id] + ' Temperature and Power')
plt.tight_layout()
plt.show()
