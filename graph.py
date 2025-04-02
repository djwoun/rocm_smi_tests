import pandas as pd
import matplotlib.pyplot as plt

# Adjust the file name if needed.
csv_file = 'papi_measurements.csv'

# Read the CSV file.
df = pd.read_csv(csv_file)

# Define your device and sensor variables.
device_id = 1
sensor_id = 2

# Construct the column names using the variables.
temperature_column = f'rocm_smi:::temp_current:device={device_id}:sensor={sensor_id}'
memory_busy_percent_column = f'rocm_smi:::memory_busy_percent:device={device_id}'
busy_percent_column = f'rocm_smi:::busy_percent:device={device_id}'

# Create the figure and the left y-axis (for Temperature)
fig, ax1 = plt.subplots(figsize=(10, 6))
l1, = ax1.plot(df['timestamp'],
               df[temperature_column],
               color='tab:red', label='Temperature')
ax1.set_xlabel('Time(s)')
ax1.set_ylabel(f'Temperature (sensor {sensor_id})(m°C)', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create the right y-axis for Busy Percent and Memory Busy Percent.
ax2 = ax1.twinx()
l2, = ax2.plot(df['timestamp'],
               df[memory_busy_percent_column],
               color='tab:blue', label='Memory Busy Percent')
l3, = ax2.plot(df['timestamp'],
               df[busy_percent_column],
               color='tab:orange', label='Busy Percent')

# Set the right y-axis label and ticks.
ax2.set_ylabel('Memory and Sub-Block Busy Percent(%)', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_yticks([20, 40, 60, 80, 100])

# Create a combined legend for all lines.
lines = [l1, l2, l3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower right')

plt.title('Time vs. Temperature (Sensor 2), Busy Percent, and Memory Busy Percent')
plt.tight_layout()
plt.show()
