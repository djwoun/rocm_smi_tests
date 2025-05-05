import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import math
import os

# --- Configuration Variables ---
figure_width = 100      # Width of the plot in inches
figure_height = 6       # Height of the plot in inches
x_tick_interval = 1     # Interval for major ticks on the x-axis (time in seconds)
temp_y_tick_interval = 5 # Interval for major ticks on the y-axis for Temperature plot (°C) - Adjusted for 0-105 range consistency
activity_y_tick_interval = 5 # <<<--- Using 5 for consistency on combined plots' activity axis
figure_dpi = 150        # Dots Per Inch (resolution - higher value = clearer image)
save_plots = True       # Set to True to save plots to files instead of just showing
output_dir = "figure/6.4.0/GEMV/58368_116736/MI210/EARLY" # Directory to save plots if save_plots is True
# --- NEW CONFIGURATION: Specify which plots to generate (by number 1-11) ---
# Example: generate all plots including the new plot 11
plots_to_generate = [1, 4, 6, 7, 8, 9, 10, 11]
# plots_to_generate = [1, 3, 5, 9, 11] # Example: generate plots 1, 3, 5, 9, and 11
# --- End Configuration ---

# Global figure size setting
FIG_SIZE = (figure_width, figure_height)

# Create output directory if saving plots and it doesn't exist
if save_plots:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to the '{output_dir}' directory.")

# Read the CSV file.
# *** IMPORTANT: Ensure this path is correct for your system ***
csv_file = "data/gilgamesh_gemv_58368_116736_128_20RUNS_Monitor_EARLY.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Convert timestamps to relative time (seconds from the start).
if 'timestamp' not in df.columns:
    print("Error: 'timestamp' column not found in CSV.")
    exit()
# Ensure timestamp is numeric before subtraction
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp']) # Remove rows where timestamp couldn't be converted
if df.empty:
    print("Error: No valid numeric timestamp data found.")
    exit()

df['time_s'] = df['timestamp'] - df['timestamp'].iloc[0]
max_time = df['time_s'].max() if not df['time_s'].empty else 0
# Calculate x-axis limit, ensuring it's a multiple of the tick interval
x_limit = math.ceil(max_time / x_tick_interval) * x_tick_interval if max_time > 0 else x_tick_interval

print(f"Data loaded. Max time: {max_time:.2f}s. Generating plots with size={FIG_SIZE}, DPI={figure_dpi}, x-tick interval={x_tick_interval}s...")
print(f"Generating plots specified in plots_to_generate: {plots_to_generate}")

# Function to show or save the plot
def display_or_save_plot(fig, filename_base):
    """Handles showing or saving the current plot."""
    if save_plots:
        png_filename = os.path.join(output_dir, f"{filename_base}.png")
        try:
            # Use facecolor='white' to avoid potential transparency issues with some viewers
            fig.savefig(png_filename, dpi=figure_dpi, bbox_inches='tight', facecolor='white')
            print(f"Saved: {png_filename}")
        except Exception as e:
            print(f"Error saving plot {filename_base}.png: {e}")
        plt.close(fig) # Close the figure after saving to free memory
    else:
        plt.show()

# --- Plot 1: Temperatures vs. Time ---
if 1 in plots_to_generate:
    print("Generating Plot 1: Temperature Plot...")
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
    temp_cols = {
        'Temp_Edge_mC': 'Edge Temp (°C)',
        'Temp_Hotspot_Junction_mC': 'Hotspot/Junction Temp (°C)',
        'Temp_VRAM_mC': 'VRAM Temp (°C)',
        'Temp_HBM0_mC': 'HBM0 Temp (°C)',
        'Temp_HBM1_mC': 'HBM1 Temp (°C)',
        'Temp_HBM2_mC': 'HBM2 Temp (°C)',
        'Temp_HBM3_mC': 'HBM3 Temp (°C)'
    }
    for col, label in temp_cols.items():
        if col in df.columns:
            # Data is already in Celsius per user instruction, despite 'mC' suffix
            temp_data_c = pd.to_numeric(df[col], errors='coerce').fillna(0)
            ax.plot(df['time_s'], temp_data_c, label=label)
        else:
            print(f"Warning (Plot 1): Column '{col}' not found.")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('GPU Temperatures over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, x_limit)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
    ax.set_ylim(0, 100) # Keeping 0-100 limit for this specific plot
    # Use the temp variable for Y-axis major ticks
    ax.yaxis.set_major_locator(mticker.MultipleLocator(temp_y_tick_interval))
    # Calculate minor tick interval based on major, ensuring it's at least 1
    temp_y_minor_tick = max(1, temp_y_tick_interval / 5)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(temp_y_minor_tick))
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    display_or_save_plot(fig, "1_gpu_temperatures") # Added number prefix to filename

# --- Plot 2: Activity Percentages vs. Time ---
if 2 in plots_to_generate:
    print("Generating Plot 2: Activity Plot...")
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
    activity_cols = {
        'GfxActivity_%': 'Graphics Activity (%)',
        'UmcActivity_%': 'Unified Memory Controller Activity (%)',
        'MmActvity_%': 'Multimedia Engine Activity (%)' # Keeping original potential typo key
    }
    mm_activity_col_actual = 'MmActvity_%'
    # Handle potential typo in column name (MmActvity vs MmActivity)
    if mm_activity_col_actual not in df.columns and 'MmActivity_%' in df.columns:
        print("Info (Plot 2): Found 'MmActivity_%' instead of 'MmActvity_%'. Using 'MmActivity_%'.")
        mm_activity_col_actual = 'MmActivity_%'
        # Update the dictionary key if necessary
        if 'MmActvity_%' in activity_cols:
            activity_cols[mm_activity_col_actual] = activity_cols.pop('MmActvity_%')
            activity_cols['MmActivity_%'] = 'Multimedia Engine Activity (%)' # Explicitly set the label again if needed

    elif mm_activity_col_actual not in df.columns and 'MmActivity_%' not in df.columns:
         print(f"Warning (Plot 2): Column '{mm_activity_col_actual}' or 'MmActivity_%' not found.")
         # Remove the entry from the dict if neither exists to avoid error later
         if mm_activity_col_actual in activity_cols:
           del activity_cols[mm_activity_col_actual]


    for col, label in activity_cols.items():
        if col in df.columns:
            activity_data = pd.to_numeric(df[col], errors='coerce').fillna(0)
            ax.plot(df['time_s'], activity_data, label=label)
        else:
             # This case should be handled by the check above for MmActvity/MmActivity
             # but kept as a fallback for other activity columns if added later.
            print(f"Warning (Plot 2): Column '{col}' not found.")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activity (%)')
    ax.set_title('GPU Activity over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, x_limit)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
    ax.set_ylim(0, 105)
    # Use the activity variable for Y-axis major ticks
    ax.yaxis.set_major_locator(mticker.MultipleLocator(activity_y_tick_interval))
    # Calculate minor tick interval based on major, ensuring it's at least 1
    activity_y_minor_tick = max(1, activity_y_tick_interval / 5)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(activity_y_minor_tick))
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    display_or_save_plot(fig, "2_gpu_activity") # Added number prefix

# --- Plot 3: Power Consumption vs. Time ---
if 3 in plots_to_generate:
    print("Generating Plot 3: Power Plot...")
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
    power_cols = {'AvgSocketPower_W': 'Average Socket Power (W)'}
    max_power_p3 = 0
    plotted_p3 = False
    for col, label in power_cols.items():
        if col in df.columns:
            power_data = pd.to_numeric(df[col], errors='coerce').fillna(0)
            if not power_data.empty:
                max_power_p3 = max(max_power_p3, power_data.max())
                ax.plot(df['time_s'], power_data, label=label)
                plotted_p3 = True
        else:
            print(f"Warning (Plot 3): Column '{col}' not found.")

    if plotted_p3:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        ax.set_title('GPU Power Consumption over Time')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_xlim(0, x_limit)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
        # Dynamic Y-axis for Power
        y_major_tick_power_p3 = 25 # Default if no data or max is 0
        if max_power_p3 > 0:
            # Calculate a reasonable major tick interval (e.g., aim for ~10-15 ticks)
            y_major_tick_power_p3 = max(10, math.ceil(max_power_p3 / 15 / 10) * 10)
        y_minor_tick_power_p3 = max(2, y_major_tick_power_p3 / 5)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(y_major_tick_power_p3))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor_tick_power_p3))
        if max_power_p3 > 0:
            # Set top limit slightly above max, aligned to the next major tick
            top_limit_power_p3 = math.ceil((max_power_p3 * 1.05) / y_major_tick_power_p3) * y_major_tick_power_p3
            ax.set_ylim(bottom=0, top=max(y_major_tick_power_p3, top_limit_power_p3)) # Ensure top isn't less than one tick
        else:
            ax.set_ylim(bottom=0, top=100) # Default reasonable limit if no data
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout(rect=[0, 0, 0.95, 1])
        display_or_save_plot(fig, "3_gpu_power") # Added number prefix
    else:
        print(f"Skipping Plot 3 generation as required columns were not found or empty.")
        plt.close(fig) # Close the empty figure if nothing was plotted

# --- Plot 4: Clock Speeds vs. Time ---
if 4 in plots_to_generate:
    print("Generating Plot 4: Clock Speed Plot...")
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
    clock_cols = {
        'ActualAvgClk_GFX_MHz': 'Average Graphics Clock (MHz)',
        'ActualAvgClk_MEM_MHz': 'Average Memory Clock (MHz)'
    }
    max_clk_p4 = 0
    plotted_p4 = False
    for col, label in clock_cols.items():
        if col in df.columns:
            clk_data = pd.to_numeric(df[col], errors='coerce').fillna(0)
            if not clk_data.empty:
                 max_clk_p4 = max(max_clk_p4, clk_data.max())
                 ax.plot(df['time_s'], clk_data, label=label)
                 plotted_p4 = True
        else:
            print(f"Warning (Plot 4): Column '{col}' not found.")

    if plotted_p4:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Clock Speed (MHz)')
        ax.set_title('GPU Clock Speeds over Time')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_xlim(0, x_limit)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
        # Dynamic Y-axis for Clock
        y_major_tick_clk_p4 = 100 # Default if no data
        if max_clk_p4 > 0:
            y_major_tick_clk_p4 = max(50, math.ceil(max_clk_p4 / 20 / 50) * 50) # Aim for ~15-20 ticks, multiple of 50
        y_minor_tick_clk_p4 = max(10, y_major_tick_clk_p4 / 5)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(y_major_tick_clk_p4))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor_tick_clk_p4))
        if max_clk_p4 > 0:
            top_limit_clk_p4 = math.ceil((max_clk_p4 * 1.05) / y_major_tick_clk_p4) * y_major_tick_clk_p4
            ax.set_ylim(bottom=0, top=max(y_major_tick_clk_p4, top_limit_clk_p4))
        else:
            ax.set_ylim(bottom=0, top=1000) # Default limit
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout(rect=[0, 0, 0.95, 1])
        display_or_save_plot(fig, "4_gpu_clocks") # Added number prefix
    else:
        print(f"Skipping Plot 4 generation as required columns were not found or empty.")
        plt.close(fig) # Close the empty figure

# --- Plot 5: Memory Usage vs. Time ---
if 5 in plots_to_generate:
    print("Generating Plot 5: VRAM Usage Plot...")
    vram_used_col = 'VramUsed_Bytes'
    vram_total_col = 'VramTotal_Bytes'
    if vram_used_col in df.columns and vram_total_col in df.columns:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi) # Create figure only if columns exist
        df[vram_used_col] = pd.to_numeric(df[vram_used_col], errors='coerce').fillna(0)
        df[vram_total_col] = pd.to_numeric(df[vram_total_col], errors='coerce').fillna(0)
        df['VramUsed_GB'] = df[vram_used_col] / (1024**3)
        df['VramTotal_GB'] = df[vram_total_col] / (1024**3)

        total_vram_gb_p5 = 0
        # Get total VRAM - try first row, fallback to max if first row is 0
        if not df['VramTotal_GB'].empty:
            total_vram_gb_p5 = df['VramTotal_GB'].iloc[0]
            if total_vram_gb_p5 == 0 :
                total_vram_gb_p5 = df['VramTotal_GB'].max()

        ax.plot(df['time_s'], df['VramUsed_GB'], label=f'Used VRAM (GB) (Total: {total_vram_gb_p5:.2f} GB)')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('VRAM Usage (GB)')
        ax.set_title('GPU VRAM Usage over Time')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_xlim(0, x_limit)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))

        # Dynamic Y-axis for VRAM
        top_limit_vram_p5 = 1 # Default
        y_major_tick_vram_p5 = 0.5 # Default
        max_used_gb_p5 = df['VramUsed_GB'].max() if not df['VramUsed_GB'].empty else 0

        if total_vram_gb_p5 > 0:
            # Set limit based on total VRAM, but ensure it also covers max used
            top_limit_vram_p5 = max(math.ceil(total_vram_gb_p5 * 1.05) , math.ceil(max_used_gb_p5 * 1.10))
            # Aim for ~10-16 ticks, multiple of 0.5
            y_major_tick_vram_p5 = max(0.5, math.ceil(top_limit_vram_p5 / 16 / 0.5) * 0.5)
        elif max_used_gb_p5 > 0: # If total VRAM unknown, base on max used
            top_limit_vram_p5 = math.ceil(max_used_gb_p5 * 1.10)
            y_major_tick_vram_p5 = max(0.5, math.ceil(top_limit_vram_p5 / 16 / 0.5) * 0.5)

        # Ensure limit is at least one major tick interval
        top_limit_vram_p5 = max(top_limit_vram_p5, y_major_tick_vram_p5)
        y_minor_tick_vram_p5 = max(0.1, y_major_tick_vram_p5 / 5)

        ax.set_ylim(bottom=0, top=top_limit_vram_p5)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(y_major_tick_vram_p5))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor_tick_vram_p5))

        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout(rect=[0, 0, 0.95, 1])
        display_or_save_plot(fig, "5_gpu_vram") # Added number prefix
    else:
        print(f"Warning (Plot 5): Columns '{vram_used_col}' or '{vram_total_col}' not found. Skipping VRAM plot.")

# --- Plot 6: Combined Temp/Activity Plot (Hotspot) ---
if 6 in plots_to_generate:
    print("Generating Plot 6: Combined Hotspot Temp/Activity Plot...")
    temp_col_p6 = 'Temp_Hotspot_Junction_mC'
    gfx_activity_col_p6 = 'GfxActivity_%'
    umc_activity_col_p6 = 'UmcActivity_%'
    required_cols_p6 = [temp_col_p6, gfx_activity_col_p6, umc_activity_col_p6]
    if all(col in df.columns for col in required_cols_p6):
        fig, ax1 = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
        ax2 = ax1.twinx() # Create a second y-axis sharing the same x-axis

        # Left Axis (Temp)
        temp_label_p6 = 'Hotspot Temp (°C)'
        # Data is already in Celsius per user instruction
        temp_data_p6 = pd.to_numeric(df[temp_col_p6], errors='coerce').fillna(0)
        l1, = ax1.plot(df['time_s'], temp_data_p6, color='tab:red', label=temp_label_p6)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(temp_label_p6, color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True, linestyle='--', linewidth=0.5) # Grid associated with ax1
        ax1.set_ylim(0, 105) # MODIFIED limit for alignment with Activity (0-105)
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(temp_y_tick_interval)) # MODIFIED ticks for 0-105 range (using 5)
        ax1.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, temp_y_tick_interval / 5))) # Match major ticks

        # Right Axis (Activity)
        gfx_activity_label_p6 = 'Graphics Activity (%)'
        umc_activity_label_p6 = 'Memory Activity (%)'
        gfx_activity_data_p6 = pd.to_numeric(df[gfx_activity_col_p6], errors='coerce').fillna(0)
        umc_activity_data_p6 = pd.to_numeric(df[umc_activity_col_p6], errors='coerce').fillna(0)
        l2, = ax2.plot(df['time_s'], gfx_activity_data_p6, color='tab:orange', label=gfx_activity_label_p6)
        l3, = ax2.plot(df['time_s'], umc_activity_data_p6, color='tab:blue', label=umc_activity_label_p6)
        ax2.set_ylabel('Activity (%)', color='black') # Label for the right axis
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 105) # Keep 0-105 limit for Activity
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(activity_y_tick_interval)) # Use activity interval (using 5)
        ax2.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, activity_y_tick_interval / 5))) # Match major ticks

        # Shared X / Legend / Title / Layout / Save
        ax1.set_xlim(0, x_limit)
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        ax1.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
        ax1.tick_params(axis='x', rotation=45)

        # Combine legends from both axes
        lines = [l1, l2, l3]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0)) # Position legend outside plot area

        plt.title('Time vs. Hotspot Temp, Graphics Activity, and UMC Activity')
        fig.tight_layout(rect=[0, 0, 0.93, 1]) # Adjust layout to make space for legend
        display_or_save_plot(fig, "6_gpu_combined_hotspot_temp_activity") # Added number prefix
    else:
        missing_cols = [col for col in required_cols_p6 if col not in df.columns]
        print(f"Warning (Plot 6): Missing columns required: {missing_cols}. Skipping.")


# --- Plot 7: Combined Temp/Activity Plot (Edge) ---
if 7 in plots_to_generate:
    print("Generating Plot 7: Combined Edge Temp/Activity Plot...")
    temp_col_p7 = 'Temp_Edge_mC'
    gfx_activity_col_p7 = 'GfxActivity_%'
    umc_activity_col_p7 = 'UmcActivity_%'
    required_cols_p7 = [temp_col_p7, gfx_activity_col_p7, umc_activity_col_p7]
    if all(col in df.columns for col in required_cols_p7):
        fig, ax1 = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
        ax2 = ax1.twinx()

        # Left Axis (Temp)
        temp_label_p7 = 'Edge Temp (°C)'
        # Data is already in Celsius per user instruction
        temp_data_p7 = pd.to_numeric(df[temp_col_p7], errors='coerce').fillna(0)
        l1, = ax1.plot(df['time_s'], temp_data_p7, color='tab:red', label=temp_label_p7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(temp_label_p7, color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.set_ylim(0, 105) # MODIFIED limit for alignment
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(temp_y_tick_interval)) # MODIFIED ticks for 0-105 range (using 5)
        ax1.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, temp_y_tick_interval / 5))) # Match major ticks

        # Right Axis (Activity)
        gfx_activity_label_p7 = 'Graphics Activity (%)'
        umc_activity_label_p7 = 'Memory Activity (%)'
        gfx_activity_data_p7 = pd.to_numeric(df[gfx_activity_col_p7], errors='coerce').fillna(0)
        umc_activity_data_p7 = pd.to_numeric(df[umc_activity_col_p7], errors='coerce').fillna(0)
        l2, = ax2.plot(df['time_s'], gfx_activity_data_p7, color='tab:orange', label=gfx_activity_label_p7)
        l3, = ax2.plot(df['time_s'], umc_activity_data_p7, color='tab:blue', label=umc_activity_label_p7)
        ax2.set_ylabel('Activity (%)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 105) # Keep 0-105 limit
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(activity_y_tick_interval)) # Use activity interval (using 5)
        ax2.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, activity_y_tick_interval / 5))) # Match major ticks

        # Shared X / Legend / Title / Layout / Save
        ax1.set_xlim(0, x_limit)
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        ax1.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
        ax1.tick_params(axis='x', rotation=45)
        lines = [l1, l2, l3]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        plt.title('Time vs. Edge Temp, Graphics Activity, and UMC Activity')
        fig.tight_layout(rect=[0, 0, 0.93, 1])
        display_or_save_plot(fig, "7_gpu_combined_edge_temp_activity") # Added number prefix
    else:
        missing_cols = [col for col in required_cols_p7 if col not in df.columns]
        print(f"Warning (Plot 7): Missing columns required: {missing_cols}. Skipping.")


# --- Plot 8: Combined Temp/Activity Plot (VRAM) ---
if 8 in plots_to_generate:
    print("Generating Plot 8: Combined VRAM Temp/Activity Plot...")
    temp_col_p8 = 'Temp_VRAM_mC'
    gfx_activity_col_p8 = 'GfxActivity_%' # Kept for check, but not plotted by default
    umc_activity_col_p8 = 'UmcActivity_%'
    required_cols_p8 = [temp_col_p8, umc_activity_col_p8] # Removed gfx_activity from required for plotting UMC
    # required_cols_p8 = [temp_col_p8, gfx_activity_col_p8, umc_activity_col_p8] # If you want to plot GFX Activity too

    if all(col in df.columns for col in required_cols_p8):
        fig, ax1 = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
        ax2 = ax1.twinx()

        # Left Axis (Temp)
        temp_label_p8 = 'VRAM Temp (°C)'
        # Data is already in Celsius per user instruction
        temp_data_p8 = pd.to_numeric(df[temp_col_p8], errors='coerce').fillna(0)
        l1, = ax1.plot(df['time_s'], temp_data_p8, color='tab:red', label=temp_label_p8)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(temp_label_p8, color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.set_ylim(0, 105) # MODIFIED limit for alignment
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(temp_y_tick_interval)) # MODIFIED ticks for 0-105 range (using 5)
        ax1.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, temp_y_tick_interval / 5))) # Match major ticks

        # Right Axis (Activity)
        umc_activity_label_p8 = 'Memory Activity (%)'
        umc_activity_data_p8 = pd.to_numeric(df[umc_activity_col_p8], errors='coerce').fillna(0)
        l3, = ax2.plot(df['time_s'], umc_activity_data_p8, color='tab:blue', label=umc_activity_label_p8)

        # --- Optional: Uncomment to plot GFX activity as well ---
        # gfx_activity_label_p8 = 'Graphics Activity (%)'
        # if gfx_activity_col_p8 in df.columns: # Check if GFX column exists
        #     gfx_activity_data_p8 = pd.to_numeric(df[gfx_activity_col_p8], errors='coerce').fillna(0)
        #     l2, = ax2.plot(df['time_s'], gfx_activity_data_p8, color='tab:orange', label=gfx_activity_label_p8)
        # else:
        #     l2 = None # Assign None if not plotted, for legend handling
        #     print(f"Warning (Plot 8): Column '{gfx_activity_col_p8}' not found, cannot plot GFX Activity.")
        # --- End Optional GFX Plot ---

        ax2.set_ylabel('Activity (%)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 105) # Keep 0-105 limit
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(activity_y_tick_interval)) # Use activity interval (using 5)
        ax2.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, activity_y_tick_interval / 5))) # Match major ticks

        # Shared X / Legend / Title / Layout / Save
        ax1.set_xlim(0, x_limit)
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        ax1.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
        ax1.tick_params(axis='x', rotation=45)

        # Adjust lines included in legend based on what's plotted
        lines = [l1, l3] # Default: VRAM Temp and UMC Activity
        # if l2: # If GFX activity was plotted successfully
        #   lines.insert(1, l2) # Insert GFX line in the middle: [l1, l2, l3]

        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        plt.title('Time vs. VRAM Temp and UMC Activity') # Adjust title if GFX added
        fig.tight_layout(rect=[0, 0, 0.93, 1])
        display_or_save_plot(fig, "8_gpu_combined_vram_temp_activity") # Added number prefix
    else:
        missing_cols = [col for col in required_cols_p8 if col not in df.columns]
        print(f"Warning (Plot 8): Missing columns required: {missing_cols}. Skipping.")


# --- Plot 9: Combined Junction Temp/Power Plot ---
if 9 in plots_to_generate:
    print("Generating Plot 9: Combined Junction Temp/Power Plot...")
    temp_col_p9 = 'Temp_Hotspot_Junction_mC'
    power_col_p9 = 'AvgSocketPower_W'
    required_cols_p9 = [temp_col_p9, power_col_p9]

    if all(col in df.columns for col in required_cols_p9):
        fig, ax1 = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
        ax2 = ax1.twinx() # ax2 is for Power

        # Left Axis (Temperature)
        temp_label_p9 = 'Hotspot/Junction Temp (°C)'
        # Data is already in Celsius per user instruction
        temp_data_p9 = pd.to_numeric(df[temp_col_p9], errors='coerce').fillna(0)
        l1, = ax1.plot(df['time_s'], temp_data_p9, color='tab:red', label=temp_label_p9)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(temp_label_p9, color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.set_ylim(0, 100) # Keeping 0-100 limit for Temp axis here (as requested in original comments)
        # Use the original temperature y-tick interval for this plot
        temp_y_tick_interval_orig = 5 # Re-state or use original variable
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(temp_y_tick_interval_orig))
        temp_y_minor_tick_p9 = max(1, temp_y_tick_interval_orig / 5)
        ax1.yaxis.set_minor_locator(mticker.MultipleLocator(temp_y_minor_tick_p9))

        # Right Axis (Power)
        power_label_p9 = 'Average Socket Power (W)'
        power_data_p9 = pd.to_numeric(df[power_col_p9], errors='coerce').fillna(0)
        max_power_p9 = power_data_p9.max() if not power_data_p9.empty else 0
        l2, = ax2.plot(df['time_s'], power_data_p9, color='tab:blue', label=power_label_p9) # Changed color for contrast
        ax2.set_ylabel(power_label_p9, color='tab:blue') # Match color
        ax2.tick_params(axis='y', labelcolor='tab:blue') # Match color

        # Dynamic Ylim and Ticks for Power axis (same logic as Plot 3)
        y_major_tick_power_p9 = 25
        if max_power_p9 > 0:
            y_major_tick_power_p9 = max(10, math.ceil(max_power_p9 / 15 / 10) * 10)
        y_minor_tick_power_p9 = max(2, y_major_tick_power_p9 / 5)
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(y_major_tick_power_p9))
        ax2.yaxis.set_minor_locator(mticker.MultipleLocator(y_minor_tick_power_p9))
        if max_power_p9 > 0:
            top_limit_power_p9 = math.ceil((max_power_p9 * 1.05) / y_major_tick_power_p9) * y_major_tick_power_p9
            ax2.set_ylim(bottom=0, top=max(y_major_tick_power_p9, top_limit_power_p9))
        else:
            ax2.set_ylim(bottom=0, top=100)

        # Shared X / Legend / Title / Layout / Save
        ax1.set_xlim(0, x_limit)
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        ax1.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
        ax1.tick_params(axis='x', rotation=45)
        lines = [l1, l2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        plt.title('Time vs. Junction Temp and Average Socket Power')
        fig.tight_layout(rect=[0, 0, 0.93, 1])
        display_or_save_plot(fig, "9_gpu_combined_junction_temp_power") # Added number prefix
    else:
        missing_cols = [col for col in required_cols_p9 if col not in df.columns]
        print(f"Warning (Plot 9): Missing columns required: {missing_cols}. Skipping.")

# --- Plot 10: Combined VRAM/HBM Temp & Memory Activity Plot ---
if 10 in plots_to_generate:
    print("Generating Plot 10: Combined VRAM/HBM Temp/Memory Activity Plot...")
    hbm_temp_cols_p10 = ['Temp_HBM0_mC', 'Temp_HBM1_mC', 'Temp_HBM2_mC', 'Temp_HBM3_mC']
    vram_temp_col_p10 = 'Temp_VRAM_mC'
    mem_activity_col_p10 = 'UmcActivity_%'
    # Update required columns list - Check all HBM temps exist individually later
    required_cols_p10 = [vram_temp_col_p10, mem_activity_col_p10] + hbm_temp_cols_p10 # Check existence of all

    if all(col in df.columns for col in required_cols_p10):
        fig, ax1 = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
        ax2 = ax1.twinx() # ax2 is for Memory Activity

        lines = [] # To collect lines for the legend

        # Left Axis (VRAM & HBM Temperatures)
        # Assign colors: one for VRAM, then cycle through others for HBM
        temp_colors = ['tab:purple', 'tab:red', 'tab:pink', 'tab:brown', 'tab:gray']
        ax1.set_ylabel('VRAM/HBM Temperature (°C)', color='black') # Updated label
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0, 105) # Use 0-105 limit for Temp vs Activity alignment
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(temp_y_tick_interval)) # Using 5
        ax1.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, temp_y_tick_interval / 5))) # Match major
        ax1.grid(True, linestyle='--', linewidth=0.5)

        # Plot VRAM Temp first
        vram_temp_label_p10 = 'VRAM Temp (°C)'
        vram_temp_data_p10 = pd.to_numeric(df[vram_temp_col_p10], errors='coerce').fillna(0)
        l_vram, = ax1.plot(df['time_s'], vram_temp_data_p10, color=temp_colors[0], label=vram_temp_label_p10, linewidth=2) # Thicker line maybe
        lines.append(l_vram)

        # Plot HBM Temps (check each column again just in case, although already checked in 'all')
        hbm_plotted_count = 0
        for i, col in enumerate(hbm_temp_cols_p10):
           if col in df.columns:
               temp_label_p10 = f'HBM{i} Temp (°C)'
               # Data is already in Celsius per user instruction
               temp_data_p10 = pd.to_numeric(df[col], errors='coerce').fillna(0)
               # Use remaining colors, cycling if necessary (index i+1)
               l, = ax1.plot(df['time_s'], temp_data_p10, color=temp_colors[(i + 1) % len(temp_colors)], label=temp_label_p10, alpha=0.8) # Slightly transparent HBM
               lines.append(l)
               hbm_plotted_count += 1
           # else: # This branch is unlikely due to the 'all' check above
           #     print(f"Warning (Plot 10): HBM Temp column '{col}' unexpectedly missing during plotting.")

        # Right Axis (Memory Activity)
        mem_activity_label_p10 = 'Memory Activity (%)'
        mem_activity_data_p10 = pd.to_numeric(df[mem_activity_col_p10], errors='coerce').fillna(0)
        l_mem, = ax2.plot(df['time_s'], mem_activity_data_p10, color='tab:blue', label=mem_activity_label_p10, linestyle='-')
        lines.append(l_mem)
        ax2.set_ylabel(mem_activity_label_p10, color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylim(0, 105) # Use 0-105 limit for Activity
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(activity_y_tick_interval)) # Using 5
        ax2.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, activity_y_tick_interval / 5))) # Match major

        # Shared X / Legend / Title / Layout / Save
        ax1.set_xlim(0, x_limit)
        ax1.set_xlabel('Time (s)')
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        ax1.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
        ax1.tick_params(axis='x', rotation=45)

        # Combine legends from both axes
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        plt.title('Time vs. VRAM/HBM Temperatures and Memory (UMC) Activity') # Updated title
        fig.tight_layout(rect=[0, 0, 0.93, 1])
        display_or_save_plot(fig, "10_gpu_combined_vram_hbm_temp_mem_activity") # Added number prefix
    else:
        missing_cols = [col for col in required_cols_p10 if col not in df.columns]
        print(f"Warning (Plot 10): Missing columns required: {missing_cols}. Skipping.")

# --- Plot 11: Combined Junction/Memory Temp & Graphics/Memory Activity Plot ---
if 11 in plots_to_generate:
    print("Generating Plot 11: Combined Junction/Memory Temp & Graphics/Memory Activity Plot...")
    junction_temp_col_p11 = 'Temp_Hotspot_Junction_mC'
    mem_temp_col_p11 = 'Temp_VRAM_mC'
    gfx_activity_col_p11 = 'GfxActivity_%'
    mem_activity_col_p11 = 'UmcActivity_%'
    required_cols_p11 = [junction_temp_col_p11, mem_temp_col_p11, gfx_activity_col_p11, mem_activity_col_p11]

    if all(col in df.columns for col in required_cols_p11):
        fig, ax1 = plt.subplots(figsize=FIG_SIZE, dpi=figure_dpi)
        ax2 = ax1.twinx() # ax2 is for Activity

        lines = [] # To collect lines for the legend

        # Left Axis (Temperatures)
        ax1.set_ylabel('Temperature (°C)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0, 105) # Use 0-105 limit for Temp vs Activity alignment
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(temp_y_tick_interval)) # Using 5
        ax1.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, temp_y_tick_interval / 5))) # Match major
        ax1.grid(True, linestyle='--', linewidth=0.5, axis='y') # Grid only on temp axis if desired

        # Plot Junction Temp
        junction_temp_label_p11 = 'Junction Temp (°C)'
        # Data is already in Celsius per user instruction
        junction_temp_data_p11 = pd.to_numeric(df[junction_temp_col_p11], errors='coerce').fillna(0)
        l_junc_temp, = ax1.plot(df['time_s'], junction_temp_data_p11, color='tab:red', label=junction_temp_label_p11)
        lines.append(l_junc_temp)

        # Plot Memory Temp
        mem_temp_label_p11 = 'Memory Temp (°C)'
        # Data is already in Celsius per user instruction
        mem_temp_data_p11 = pd.to_numeric(df[mem_temp_col_p11], errors='coerce').fillna(0)
        l_mem_temp, = ax1.plot(df['time_s'], mem_temp_data_p11, color='tab:purple', label=mem_temp_label_p11)
        lines.append(l_mem_temp)

        # Right Axis (Activity)
        ax2.set_ylabel('Activity (%)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(0, 105) # Use 0-105 limit for Activity
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(activity_y_tick_interval)) # Using 5
        ax2.yaxis.set_minor_locator(mticker.MultipleLocator(max(1, activity_y_tick_interval / 5))) # Match major

        # Plot Graphics Activity
        gfx_activity_label_p11 = 'Graphics Activity (%)'
        gfx_activity_data_p11 = pd.to_numeric(df[gfx_activity_col_p11], errors='coerce').fillna(0)
        l_gfx_act, = ax2.plot(df['time_s'], gfx_activity_data_p11, color='tab:orange', label=gfx_activity_label_p11, linestyle='-')
        lines.append(l_gfx_act)

        # Plot Memory Activity
        mem_activity_label_p11 = 'Memory Activity (%)'
        mem_activity_data_p11 = pd.to_numeric(df[mem_activity_col_p11], errors='coerce').fillna(0)
        l_mem_act, = ax2.plot(df['time_s'], mem_activity_data_p11, color='tab:blue', label=mem_activity_label_p11, linestyle='-')
        lines.append(l_mem_act)


        # Shared X / Legend / Title / Layout / Save
        ax1.set_xlim(0, x_limit)
        ax1.set_xlabel('Time (s)')
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        ax1.xaxis.set_minor_locator(mticker.MultipleLocator(max(1, x_tick_interval / 5)))
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', linewidth=0.5, axis='x') # Add x-axis grid lines controlled by ax1

        # Combine legends from both axes
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        plt.title('Time vs. Junction/Memory Temperature and Graphics/Memory Activity') # Updated title
        fig.tight_layout(rect=[0, 0, 0.93, 1]) # Adjust layout for legend
        display_or_save_plot(fig, "11_gpu_combined_junc_mem_temp_gfx_mem_activity") # Added number prefix
    else:
        missing_cols = [col for col in required_cols_p11 if col not in df.columns]
        print(f"Warning (Plot 11): Missing columns required: {missing_cols}. Skipping.")


print("Finished generating selected plots.")