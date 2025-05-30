import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Adjust the file name if needed.
csv_file = 'data/odyssey_gemm_3841_524288_1024_11RUNS.csv'

# Read the CSV file.
df = pd.read_csv(csv_file)

# Convert timestamps to relative time (seconds from the start).
df['time_s'] = df['timestamp'] - df['timestamp'].iloc[0]

# Define your device and sensor variables.
device_id = 1
sensor_id = 2

temp = ["Edge", "Junction", "Memory"]
temperature_column = f'rocm_smi:::temp_current:device={device_id}:sensor={sensor_id}'
memory_busy_percent_column = f'rocm_smi:::memory_busy_percent:device={device_id}'
busy_percent_column = f'rocm_smi:::busy_percent:device={device_id}'

# Create the figure and the left y-axis (for Temperature).
# Use a more moderate figure size so the data is easier to inspect.
fig, ax1 = plt.subplots(figsize=(90, 6))

# Plot temperature on the left axis.
l1, = ax1.plot(df['time_s'],
               df[temperature_column] / 1000,
               color='tab:red',
               label=temp[sensor_id] + ' Temperature')

ax1.set_xlabel('Time (s)')  # now it's relative seconds
ax1.set_ylabel(temp[sensor_id] + ' Temperature (°C)', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Add horizontal grid lines for readability.
ax1.yaxis.grid(True, linestyle='--', linewidth=0.5)

# Improve the x-axis tick placement to show more detail.
# Increase the number of major ticks so we can better see each iteration.
ax1.set_xticks(range(0, 3600, 5))


# Create the right y-axis for Busy Percent and Memory Busy Percent.
ax2 = ax1.twinx()
l2, = ax2.plot(df['time_s'],
               df[memory_busy_percent_column],
               color='tab:blue',
               label='Memory Busy Percent')
l3, = ax2.plot(df['time_s'],
               df[busy_percent_column],
               color='tab:orange',
               label='Busy Percent')

ax2.set_ylabel('Memory and Sub-Block Busy Percent (%)', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0, 100)
ax2.set_yticks(range(0, 101, 5))

ax2.set_ylim(0, 100)
ax1.set_yticks(range(0, 101, 5))
ax1.xaxis.grid(True, linestyle='--', linewidth=0.5)

# Create a combined legend for all lines.
lines = [l1, l2, l3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 0.9))

plt.title('Time vs. ' + temp[sensor_id] + ' Temperature, Busy Percent, and Memory Busy Percent')
plt.tight_layout()
plt.show()
