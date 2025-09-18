#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"
#include <unistd.h>      // For usleep()
#include <sys/time.h>    // For gettimeofday()
#include <pthread.h>     // For pthreads
#include <amd_smi/amdsmi.h> // Include AMD SMI header
#include <stdint.h>      // Include for int64_t, uint64_t, uint32_t type
#include <string.h>      // For strstr, memset, memcpy

#define M_DIM 14592
#define K_DIM 65536
#define N_DIM 14592 //14592

// Number of streams to use for concurrent execution
#define NUM_STREAMS 1

// Number of iterations to run in each stream
#define ITERATIONS_PER_STREAM 1

// Global flag to signal the monitor thread to stop.
volatile int stop_monitor = 0;

// --- AMD SMI Temperature Sensor Definitions ---
typedef struct {
    amdsmi_temperature_type_t type;
    const char* name;
} temp_sensor_info_t;

temp_sensor_info_t amd_smi_sensors[] = {
    {AMDSMI_TEMPERATURE_TYPE_EDGE,    "Temp_Edge"},
    {AMDSMI_TEMPERATURE_TYPE_HOTSPOT, "Temp_Hotspot_Junction"},
    {AMDSMI_TEMPERATURE_TYPE_VRAM,    "Temp_VRAM"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_0,   "Temp_HBM0"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_1,   "Temp_HBM1"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_2,   "Temp_HBM2"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_3,   "Temp_HBM3"},
    {AMDSMI_TEMPERATURE_TYPE_PLX,     "Temp_PLX"}
};
const int NUM_AMD_SMI_SENSORS = sizeof(amd_smi_sensors) / sizeof(amd_smi_sensors[0]);

// --- AMD SMI GPU Activity Metric Definitions ---
const char* amd_smi_activity_names[] = {
    "GfxActivity_%",
    "UmcActivity_%", // Unified Memory Controller
    "MmActvity_%"    // Multimedia Engine
};
const int NUM_AMD_SMI_ACTIVITY_METRICS = sizeof(amd_smi_activity_names) / sizeof(amd_smi_activity_names[0]); // Should be 3

// --- AMD SMI VRAM Usage Metric Definitions ---
const char* amd_smi_vram_names[] = {
    "VramTotal_Bytes", // Updated name suffix to reflect source units
    "VramUsed_Bytes"   // Updated name suffix to reflect source units
};
const int NUM_AMD_SMI_VRAM_METRICS = sizeof(amd_smi_vram_names) / sizeof(amd_smi_vram_names[0]); // Should be 2

// --- AMD SMI Power Info Metric Definitions ---
const char* amd_smi_power_info_names[] = {
    "AvgSocketPower_W",
    "CurrentSocketPower_W", // <-- ADDED
    "GfxVoltage_mV",
    "SocVoltage_mV",
    "MemVoltage_mV",
    "PowerLimit_W"
};
const int NUM_AMD_SMI_POWER_INFO_METRICS = sizeof(amd_smi_power_info_names) / sizeof(amd_smi_power_info_names[0]); // Should be 6 now

// --- AMD SMI Clock Frequency Metric Definitions (Using amdsmi_get_clk_freq) ---
typedef struct {
    amdsmi_clk_type_t type;
    const char* name;
} clk_freq_def_t;

clk_freq_def_t amd_smi_clocks[] = {
    {AMDSMI_CLK_TYPE_SYS, "SelectedClk_SYS_MHz"}, // System clock, often represents GPU clock
    {AMDSMI_CLK_TYPE_MEM, "SelectedClk_MEM_MHz"}, // Memory clock
    {AMDSMI_CLK_TYPE_DF,  "SelectedClk_DF_MHz"},  // Data Fabric clock (if needed and supported)
    // {AMDSMI_CLK_TYPE_DCEF, "SelectedClk_DCEF_MHz"} // Display Controller clock (if needed and supported)
};
const int NUM_AMD_SMI_CLOCK_METRICS = sizeof(amd_smi_clocks) / sizeof(amd_smi_clocks[0]);

// --- AMD SMI Clock Info Metric Definitions (Using amdsmi_get_clock_info) ---
typedef struct {
    amdsmi_clk_type_t type;
    const char* name;
} clk_info_def_t;

clk_info_def_t amd_smi_clock_info_defs[] = {
    {AMDSMI_CLK_TYPE_GFX, "ActualAvgClk_GFX_MHz"}, // Graphics Clock
    {AMDSMI_CLK_TYPE_MEM, "ActualAvgClk_MEM_MHz"}  // Memory Clock
};
const int NUM_AMD_SMI_CLOCK_INFO_METRICS = sizeof(amd_smi_clock_info_defs) / sizeof(amd_smi_clock_info_defs[0]);

// --- AMD SMI Fan Speed Metric Definitions ---
const char* amd_smi_fan_speed_names[] = {
    "FanSpeed_RPM" // Assuming one fan sensor (index 0) for now
};
const int NUM_AMD_SMI_FAN_METRICS = sizeof(amd_smi_fan_speed_names) / sizeof(amd_smi_fan_speed_names[0]); // Should be 1

// --- AMD SMI PCIe Static Info Definitions ---
const char* amd_smi_pcie_static_names[] = {
    "PCIe_MaxWidth",
    "PCIe_MaxSpeed_GTs", // GigaTransfers per second
    "PCIe_InterfaceVersion",
    "PCIe_SlotType", // amdsmi_card_form_factor_t enum
    "PCIe_MaxInterfaceVersion"
};
const int NUM_AMD_SMI_PCIE_STATIC_METRICS = sizeof(amd_smi_pcie_static_names) / sizeof(amd_smi_pcie_static_names[0]);

// --- AMD SMI PCIe Metric Definitions ---
const char* amd_smi_pcie_metric_names[] = {
    "PCIe_CurrentWidth",
    "PCIe_CurrentSpeed_MTs", // MegaTransfers per second (NOTE: AMDSMI struct uses GT/s * 0.1)
    "PCIe_CurrentBandwidth_Mbps", // Megabits per second
    "PCIe_ReplayCount",
    "PCIe_L0ToRecoveryCount",
    "PCIe_ReplayRollOverCount",
    "PCIe_NakSentCount",
    "PCIe_NakReceivedCount",
    "PCIe_LcPerfOtherEndRecoveryCount"
};
const int NUM_AMD_SMI_PCIE_METRIC_METRICS = sizeof(amd_smi_pcie_metric_names) / sizeof(amd_smi_pcie_metric_names[0]);

// --- START: AMD SMI GPU Metrics (amdsmi_get_gpu_metrics_info) Definitions ---
const char* amd_smi_gpu_metric_names[] = {
    "throttle_status_mask",       // Raw bitmask - needs decoding
    "indep_throttle_status_mask", // Raw bitmask - needs decoding
    "prochot_residency_acc_ns",   // Accumulated ns - calculate delta
    "ppt_residency_acc_ns",       // Accumulated ns - calculate delta
    "socket_thm_residency_acc_ns",// Accumulated ns - calculate delta
    "vr_thm_residency_acc_ns",    // Accumulated ns - calculate delta
    "hbm_thm_residency_acc_ns",   // Accumulated ns - calculate delta
    "accumulation_counter_ns"     // Reference counter for residency deltas
};
const int NUM_AMD_SMI_GPU_METRICS = sizeof(amd_smi_gpu_metric_names) / sizeof(amd_smi_gpu_metric_names[0]);
// --- END: AMD SMI GPU Metrics Definitions ---


// Calculate total number of *CSV* metrics to be collected
const int TOTAL_CSV_METRICS = NUM_AMD_SMI_SENSORS
                             + NUM_AMD_SMI_ACTIVITY_METRICS
                             + NUM_AMD_SMI_VRAM_METRICS
                             + NUM_AMD_SMI_POWER_INFO_METRICS
                             + NUM_AMD_SMI_CLOCK_METRICS        // From amdsmi_get_clk_freq
                             + NUM_AMD_SMI_CLOCK_INFO_METRICS   // From amdsmi_get_clock_info
                             + NUM_AMD_SMI_FAN_METRICS          // Fan speed
                             + NUM_AMD_SMI_PCIE_STATIC_METRICS  // PCIe Static Info
                             + NUM_AMD_SMI_PCIE_METRIC_METRICS  // PCIe Dynamic Metrics
                             + NUM_AMD_SMI_GPU_METRICS;         // GPU Metrics (Throttle, Residency)


// Structure to pass parameters to the monitoring thread.
struct monitor_params {
    amdsmi_processor_handle gpu_handle;
    FILE *csvFile;
    struct timeval start_time;
    uint32_t monitor_interval_us; // Monitoring interval in microseconds
};

// --- Function Prototypes for AMD SMI Collection ---
void collect_temperatures(amdsmi_processor_handle handle, long long* output_values);
void collect_activity(amdsmi_processor_handle handle, long long* output_values);
void collect_vram(amdsmi_processor_handle handle, long long* output_values);
void collect_power(amdsmi_processor_handle handle, long long* output_values);
void collect_selected_clocks(amdsmi_processor_handle handle, long long* output_values);
void collect_average_clocks(amdsmi_processor_handle handle, long long* output_values);
void collect_fan_speed(amdsmi_processor_handle handle, long long* output_values);
void collect_pcie_info(amdsmi_processor_handle handle, long long* output_values);
void collect_gpu_metrics(amdsmi_processor_handle handle, amdsmi_gpu_metrics_t *gpu_metrics_struct, int *first_read_flag, amdsmi_gpu_metrics_t *prev_metrics_struct, long long* output_values);
amdsmi_status_t get_process_list(amdsmi_processor_handle handle, amdsmi_proc_info_t** buffer, uint32_t* allocated_size, uint32_t* num_processes_found);

// --- START: AMD SMI Collection Functions ---

void collect_temperatures(amdsmi_processor_handle handle, long long* output_values) {
    amdsmi_status_t smi_status;
    for (int i = 0; i < NUM_AMD_SMI_SENSORS; ++i) {
        int64_t current_temp_smi = -1;
        smi_status = amdsmi_get_temp_metric(
            handle,
            amd_smi_sensors[i].type,
            AMDSMI_TEMP_CURRENT,
            &current_temp_smi
        );
        output_values[i] = (smi_status == AMDSMI_STATUS_SUCCESS) ? (long long)current_temp_smi : -1;
    }
}

void collect_activity(amdsmi_processor_handle handle, long long* output_values) {
    amdsmi_status_t smi_status;
    amdsmi_engine_usage_t activity_info;
    smi_status = amdsmi_get_gpu_activity(handle, &activity_info);
    if (smi_status != AMDSMI_STATUS_SUCCESS) {
        for (int i = 0; i < NUM_AMD_SMI_ACTIVITY_METRICS; ++i) output_values[i] = -3;
    } else {
        output_values[0] = (long long)activity_info.gfx_activity;
        output_values[1] = (long long)activity_info.umc_activity;
        output_values[2] = (long long)activity_info.mm_activity;
    }
}

void collect_vram(amdsmi_processor_handle handle, long long* output_values) {
    uint64_t total_vram_bytes = 0;
    uint64_t used_vram_bytes = 0;
    amdsmi_status_t smi_status_total, smi_status_used;
    smi_status_total = amdsmi_get_gpu_memory_total(handle, AMDSMI_MEM_TYPE_VRAM, &total_vram_bytes);
    smi_status_used = amdsmi_get_gpu_memory_usage(handle, AMDSMI_MEM_TYPE_VRAM, &used_vram_bytes);
    output_values[0] = (smi_status_total == AMDSMI_STATUS_SUCCESS) ? (long long)total_vram_bytes : -5;
    output_values[1] = (smi_status_used == AMDSMI_STATUS_SUCCESS) ? (long long)used_vram_bytes : -6;
}

void collect_power(amdsmi_processor_handle handle, long long* output_values) {
    amdsmi_status_t smi_status;
    amdsmi_power_info_t power_info;
    smi_status = amdsmi_get_power_info(handle, &power_info);
    if (smi_status != AMDSMI_STATUS_SUCCESS) {
        for (int i = 0; i < NUM_AMD_SMI_POWER_INFO_METRICS; ++i) output_values[i] = -4;
    } else {
        output_values[0] = (long long)power_info.average_socket_power;
        output_values[1] = (long long)power_info.current_socket_power;
        output_values[2] = (long long)power_info.gfx_voltage;
        output_values[3] = (long long)power_info.soc_voltage;
        output_values[4] = (long long)power_info.mem_voltage;
        output_values[5] = (long long)power_info.power_limit;
    }
}

void collect_selected_clocks(amdsmi_processor_handle handle, long long* output_values) {
    amdsmi_status_t smi_status;
    amdsmi_frequencies_t freq_info;
    for (int i = 0; i < NUM_AMD_SMI_CLOCK_METRICS; ++i) {
        smi_status = amdsmi_get_clk_freq(handle, amd_smi_clocks[i].type, &freq_info);
        if (smi_status != AMDSMI_STATUS_SUCCESS) {
            output_values[i] = -7;
        } else {
            if (freq_info.current < freq_info.num_supported && freq_info.current < AMDSMI_MAX_NUM_FREQUENCIES) {
                uint64_t current_freq_hz = freq_info.frequency[freq_info.current];
                output_values[i] = (long long)(current_freq_hz / 1000000); // Convert Hz to MHz
            } else {
                output_values[i] = -8; // Invalid index error
            }
        }
    }
}

void collect_average_clocks(amdsmi_processor_handle handle, long long* output_values) {
    amdsmi_status_t smi_status;
    amdsmi_clk_info_t clock_info_data;
    for (int i = 0; i < NUM_AMD_SMI_CLOCK_INFO_METRICS; ++i) {
        smi_status = amdsmi_get_clock_info(handle, amd_smi_clock_info_defs[i].type, &clock_info_data);
        if (smi_status != AMDSMI_STATUS_SUCCESS) {
            output_values[i] = -10;
        } else {
            output_values[i] = (long long)clock_info_data.clk; // Already in MHz
        }
    }
}

void collect_fan_speed(amdsmi_processor_handle handle, long long* output_values) {
    amdsmi_status_t smi_status;
    int64_t current_fan_rpm = -1;
    for (int i = 0; i < NUM_AMD_SMI_FAN_METRICS; ++i) {
        uint32_t sensor_index = (uint32_t)i; // Assuming sensor index corresponds to loop index
        smi_status = amdsmi_get_gpu_fan_rpms(handle, sensor_index, &current_fan_rpm);
        if (smi_status != AMDSMI_STATUS_SUCCESS) {
            if (smi_status == AMDSMI_STATUS_NOT_SUPPORTED) output_values[i] = -12;
            else output_values[i] = -11;
        } else {
            output_values[i] = (long long)current_fan_rpm;
        }
    }
}

void collect_pcie_info(amdsmi_processor_handle handle, long long* output_values) {
    amdsmi_status_t smi_status;
    amdsmi_pcie_info_t pcie_info;
    smi_status = amdsmi_get_pcie_info(handle, &pcie_info);
    if (smi_status != AMDSMI_STATUS_SUCCESS) {
        // Mark all PCIe metrics as failed
        for (int i = 0; i < NUM_AMD_SMI_PCIE_STATIC_METRICS; ++i) output_values[i] = -13;
        for (int i = 0; i < NUM_AMD_SMI_PCIE_METRIC_METRICS; ++i) output_values[NUM_AMD_SMI_PCIE_STATIC_METRICS + i] = -14;
    } else {
        // Extract Static PCIe Info
        output_values[0] = (long long)pcie_info.pcie_static.max_pcie_width;
        output_values[1] = (long long)pcie_info.pcie_static.max_pcie_speed;
        output_values[2] = (long long)pcie_info.pcie_static.pcie_interface_version;
        output_values[3] = (long long)pcie_info.pcie_static.slot_type; // Enum value
        output_values[4] = (long long)pcie_info.pcie_static.pcie_interface_version;

        // Extract Dynamic PCIe Metrics (start writing after static metrics)
        int metric_base = NUM_AMD_SMI_PCIE_STATIC_METRICS;
        output_values[metric_base + 0] = (long long)pcie_info.pcie_metric.pcie_width;
        output_values[metric_base + 1] = (long long)pcie_info.pcie_metric.pcie_speed; // Units: 0.1 GT/s
        output_values[metric_base + 2] = (long long)pcie_info.pcie_metric.pcie_bandwidth; // MBps? Check docs.
        output_values[metric_base + 3] = (long long)pcie_info.pcie_metric.pcie_replay_count;
        output_values[metric_base + 4] = (long long)pcie_info.pcie_metric.pcie_l0_to_recovery_count;
        output_values[metric_base + 5] = (long long)pcie_info.pcie_metric.pcie_replay_roll_over_count;
        output_values[metric_base + 6] = (long long)pcie_info.pcie_metric.pcie_nak_sent_count;
        output_values[metric_base + 7] = (long long)pcie_info.pcie_metric.pcie_nak_received_count;
        output_values[metric_base + 8] = (long long)pcie_info.pcie_metric.pcie_lc_perf_other_end_recovery_count;
    }
}

void collect_gpu_metrics(amdsmi_processor_handle handle,
                         amdsmi_gpu_metrics_t *current_gpu_metrics, // Pass struct to fill
                         int *first_read_flag,
                         amdsmi_gpu_metrics_t *prev_gpu_metrics, // Pass struct to read previous
                         long long* output_values)
{
    amdsmi_status_t smi_status;
    smi_status = amdsmi_get_gpu_metrics_info(handle, current_gpu_metrics);

    if (smi_status != AMDSMI_STATUS_SUCCESS) {
        for (int i = 0; i < NUM_AMD_SMI_GPU_METRICS; ++i) {
            if (smi_status == AMDSMI_STATUS_NOT_SUPPORTED) output_values[i] = -16;
            else output_values[i] = -15;
        }
    } else {
        // Store the raw values directly into the output array
        output_values[0] = (long long)current_gpu_metrics->throttle_status;
        output_values[1] = (long long)current_gpu_metrics->indep_throttle_status;
        output_values[2] = (long long)current_gpu_metrics->prochot_residency_acc;
        output_values[3] = (long long)current_gpu_metrics->ppt_residency_acc;
        output_values[4] = (long long)current_gpu_metrics->socket_thm_residency_acc;
        output_values[5] = (long long)current_gpu_metrics->vr_thm_residency_acc;
        output_values[6] = (long long)current_gpu_metrics->hbm_thm_residency_acc;
        output_values[7] = (long long)current_gpu_metrics->accumulation_counter;

        // Handle first read flag - delta calculations could be done here if needed
        if (*first_read_flag) {
            *first_read_flag = 0;
        } else {
             // Delta calculation example (not modifying output_values here):
             // uint64_t delta_accum_ns = current_gpu_metrics->accumulation_counter - prev_gpu_metrics->accumulation_counter;
             // if (delta_accum_ns > 0) { ... }
        }
        // Store the current metrics as 'previous' for the next call
        memcpy(prev_gpu_metrics, current_gpu_metrics, sizeof(amdsmi_gpu_metrics_t));
    }
}


amdsmi_status_t get_process_list(amdsmi_processor_handle handle,
                                 amdsmi_proc_info_t** buffer, // Pointer to the buffer pointer
                                 uint32_t* allocated_size,    // Pointer to allocated size
                                 uint32_t* num_processes_found) // Pointer to store result count
{
    amdsmi_status_t smi_status;
    uint32_t required_size = 0;

    // First call: get the number of processes
    *num_processes_found = 0; // Reset count
    smi_status = amdsmi_get_gpu_process_list(handle, &required_size, NULL);

    if (smi_status != AMDSMI_STATUS_SUCCESS) {
        if (smi_status == AMDSMI_STATUS_NOT_SUPPORTED) {
            // Feature not supported, return status, count remains 0
            return smi_status;
        } else {
             fprintf(stderr, "Monitor Error: Failed to get process count (Error: %d)\n", smi_status);
             return smi_status; // Return error status
        }
    }

    // If no processes are running, we are done.
    if (required_size == 0) {
        *num_processes_found = 0;
        return AMDSMI_STATUS_SUCCESS;
    }

    // Check if buffer needs allocation or reallocation
    if (required_size > *allocated_size) {
        amdsmi_proc_info_t *new_buffer = (amdsmi_proc_info_t*)realloc(*buffer, required_size * sizeof(amdsmi_proc_info_t));
        if (new_buffer == NULL) {
            fprintf(stderr, "Monitor Error: Failed to reallocate memory for process list (%u processes).\n", required_size);
            free(*buffer); // Free old buffer if realloc failed
            *buffer = NULL;
            *allocated_size = 0;
            return AMDSMI_STATUS_OUT_OF_RESOURCES; // Indicate memory failure
        }
        *buffer = new_buffer;
        *allocated_size = required_size;
        // fprintf(stdout,"Reallocated proc list buffer to size %u\n", *allocated_size); // Debug print
    }

    // Second call: get the actual process list details
    // Note: The required size could potentially increase between the two calls.
    // We pass the allocated size, and the API will update it with the actual number written.
    uint32_t buffer_size_in = *allocated_size;
    smi_status = amdsmi_get_gpu_process_list(handle, &buffer_size_in, *buffer);

    if (smi_status == AMDSMI_STATUS_SUCCESS) {
         *num_processes_found = buffer_size_in; // Update with actual number returned
    } else if (smi_status == AMDSMI_STATUS_OUT_OF_RESOURCES) {
         // This indicates the list grew between calls and our buffer wasn't large enough.
         // The required size is now in buffer_size_in. We could try again, but for simplicity:
         fprintf(stderr, "Monitor Warning: Process list size increased between calls. Required: %u, Allocated: %u. Data may be incomplete for this cycle.\n", buffer_size_in, *allocated_size);
         *num_processes_found = 0; // Indicate failure for this cycle
    } else {
         fprintf(stderr, "Monitor Error: Failed to get process list details (Error: %d)\n", smi_status);
         *num_processes_found = 0; // Indicate failure
    }

    return smi_status; // Return the status of the second call
}

// --- END: AMD SMI Collection Functions ---


// Monitor thread that periodically reads AMD SMI metrics and logs them.
void *monitor_events(void *args) {
    struct monitor_params *params = (struct monitor_params *)args;
    amdsmi_status_t smi_status; // General status

    // Array holds AMD SMI temp, activity%, VRAM, power info, clock freq, clock info, FAN speed, PCIe, GPU metrics values for CSV
    long long csv_values[TOTAL_CSV_METRICS]; // Using long long

    // --- Variables specific to GPU Metrics collection ---
    amdsmi_gpu_metrics_t current_gpu_metrics; // Structure to hold current metrics
    amdsmi_gpu_metrics_t prev_gpu_metrics;    // Structure to hold previous metrics (for delta calculation if needed)
    int first_gpu_metrics_read = 1;           // Flag for first GPU metrics read
    memset(&prev_gpu_metrics, 0, sizeof(amdsmi_gpu_metrics_t)); // Initialize previous

    // --- Variables for Process List ---
    amdsmi_proc_info_t *proc_list_buffer = NULL;
    uint32_t allocated_proc_list_size = 0; // Current allocated size of buffer
    uint32_t num_processes = 0;            // Number of processes found in the current cycle
    int first_proc_call_timing_check = 1;  // Flag to check initial delay for process list

    // *** Check monitoring interval warning ***
    if (params->monitor_interval_us < 1000000) {
        fprintf(stderr, "\n*** WARNING: Monitoring interval (%.3f s) is less than 1 second. ***\n", params->monitor_interval_us / 1e6);
        fprintf(stderr, "*** AMD SMI process list data validity requires >= 1 second between calls. ***\n\n");
    }

    while (!stop_monitor) {
        // Reset CSV values array with a placeholder/error code
        for(int i=0; i < TOTAL_CSV_METRICS; ++i) {
            csv_values[i] = -99;
        }

        // --- Time Calculation ---
        struct timeval current_time;
        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - params->start_time.tv_sec) +
                         (current_time.tv_usec - params->start_time.tv_usec) / 1e6;

        // --- Collect Metrics using Helper Functions ---
        if (params->gpu_handle != NULL) {
            int current_base_index = 0;

            collect_temperatures(params->gpu_handle, &csv_values[current_base_index]);
            current_base_index += NUM_AMD_SMI_SENSORS;

            collect_activity(params->gpu_handle, &csv_values[current_base_index]);
            current_base_index += NUM_AMD_SMI_ACTIVITY_METRICS;

            collect_vram(params->gpu_handle, &csv_values[current_base_index]);
            current_base_index += NUM_AMD_SMI_VRAM_METRICS;

            collect_power(params->gpu_handle, &csv_values[current_base_index]);
            current_base_index += NUM_AMD_SMI_POWER_INFO_METRICS;

            collect_selected_clocks(params->gpu_handle, &csv_values[current_base_index]);
            current_base_index += NUM_AMD_SMI_CLOCK_METRICS;

            collect_average_clocks(params->gpu_handle, &csv_values[current_base_index]);
            current_base_index += NUM_AMD_SMI_CLOCK_INFO_METRICS;

            collect_fan_speed(params->gpu_handle, &csv_values[current_base_index]);
            current_base_index += NUM_AMD_SMI_FAN_METRICS;

            collect_pcie_info(params->gpu_handle, &csv_values[current_base_index]);
            current_base_index += NUM_AMD_SMI_PCIE_STATIC_METRICS + NUM_AMD_SMI_PCIE_METRIC_METRICS; // PCIe func handles both

            collect_gpu_metrics(params->gpu_handle, &current_gpu_metrics, &first_gpu_metrics_read, &prev_gpu_metrics, &csv_values[current_base_index]);
            current_base_index += NUM_AMD_SMI_GPU_METRICS;

            // --- Sanity Check ---
            if (current_base_index != TOTAL_CSV_METRICS) {
                fprintf(stderr, "ERROR: CSV Metric index mismatch! Expected %d, got %d\n", TOTAL_CSV_METRICS, current_base_index);
            }

            // --- Get Process List ---
            // Check timing requirement for the *very first* call
            if (first_proc_call_timing_check) {
                 struct timeval now;
                 gettimeofday(&now, NULL);
                 double time_since_start = (now.tv_sec - params->start_time.tv_sec) +
                                           (now.tv_usec - params->start_time.tv_usec) / 1e6;
                 if (time_since_start < 1.0) {
                     fprintf(stderr, "Monitor Warning: First call to get_process_list occurred %.3f seconds after start. Minimum 1 second required for valid data.\n", time_since_start);
                 }
                 first_proc_call_timing_check = 0; // Only check this once
            }

            // Call the function to get the process list
            smi_status = get_process_list(params->gpu_handle, &proc_list_buffer, &allocated_proc_list_size, &num_processes);
            // The function handles internal errors/warnings and updates num_processes accordingly.
            // smi_status contains the final status (e.g., SUCCESS, NOT_SUPPORTED, error)


        } else { // If handle is NULL
            for(int i = 0; i < TOTAL_CSV_METRICS; ++i) {
                csv_values[i] = -2; // AMD SMI handle error code
            }
            num_processes = 0; // Can't get processes without handle
            smi_status = AMDSMI_STATUS_NO_DATA; // Indicate no handle
        }


        // --- Write *CSV* Measurements to CSV ---
        fprintf(params->csvFile, "%.6f", elapsed);
        for (int i = 0; i < TOTAL_CSV_METRICS; ++i) {
            fprintf(params->csvFile, ",%lld", csv_values[i]); // Use %lld for long long
        }
        fprintf(params->csvFile, "\n");
        fflush(params->csvFile);

        // --- Print Measurements to Stdout ---
        fprintf(stdout, "Time: %.6f sec -> ", elapsed);
        int print_base_idx = 0;

        // Temps
        fprintf(stdout, "Temps[0..%d]: [", NUM_AMD_SMI_SENSORS - 1);
         for (int i = 0; i < NUM_AMD_SMI_SENSORS; ++i) fprintf(stdout, "%s=%lld%s", amd_smi_sensors[i].name, csv_values[print_base_idx + i], (i == NUM_AMD_SMI_SENSORS - 1) ? "" : ", ");
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_SENSORS;

        // Activity %
        fprintf(stdout, "Activity[0..%d]: [", NUM_AMD_SMI_ACTIVITY_METRICS -1);
        for (int i = 0; i < NUM_AMD_SMI_ACTIVITY_METRICS; ++i) fprintf(stdout, "%s=%lld%s", amd_smi_activity_names[i], csv_values[print_base_idx + i], (i == NUM_AMD_SMI_ACTIVITY_METRICS - 1) ? "" : ", ");
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_ACTIVITY_METRICS;

        // VRAM Usage
        fprintf(stdout, "VRAM[0..%d]: [", NUM_AMD_SMI_VRAM_METRICS -1);
        for (int i = 0; i < NUM_AMD_SMI_VRAM_METRICS; ++i) fprintf(stdout, "%s=%lld%s", amd_smi_vram_names[i], csv_values[print_base_idx + i], (i == NUM_AMD_SMI_VRAM_METRICS - 1) ? "" : ", ");
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_VRAM_METRICS;

        // Power Info
        fprintf(stdout, "PwrInfo[0..%d]: [", NUM_AMD_SMI_POWER_INFO_METRICS - 1);
        for (int i = 0; i < NUM_AMD_SMI_POWER_INFO_METRICS; ++i) fprintf(stdout, "%s=%lld%s", amd_smi_power_info_names[i], csv_values[print_base_idx + i], (i == NUM_AMD_SMI_POWER_INFO_METRICS - 1) ? "" : ", ");
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_POWER_INFO_METRICS;

        // Clock Frequencies (Selected Level)
        fprintf(stdout, "SelectedClocks[0..%d]: [", NUM_AMD_SMI_CLOCK_METRICS - 1);
        for (int i = 0; i < NUM_AMD_SMI_CLOCK_METRICS; ++i) fprintf(stdout, "%s=%lld%s", amd_smi_clocks[i].name, csv_values[print_base_idx + i], (i == NUM_AMD_SMI_CLOCK_METRICS - 1) ? "" : ", ");
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_CLOCK_METRICS;

        // Clock Info (Actual Average)
        fprintf(stdout, "ActualAvgClocks[0..%d]: [", NUM_AMD_SMI_CLOCK_INFO_METRICS - 1);
        for (int i = 0; i < NUM_AMD_SMI_CLOCK_INFO_METRICS; ++i) fprintf(stdout, "%s=%lld%s", amd_smi_clock_info_defs[i].name, csv_values[print_base_idx + i], (i == NUM_AMD_SMI_CLOCK_INFO_METRICS - 1) ? "" : ", ");
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_CLOCK_INFO_METRICS;

        // Fan Speed
        fprintf(stdout, "FanSpeed[0..%d]: [", NUM_AMD_SMI_FAN_METRICS - 1);
        for (int i = 0; i < NUM_AMD_SMI_FAN_METRICS; ++i) {
            fprintf(stdout, "%s=%lld%s", amd_smi_fan_speed_names[i], csv_values[print_base_idx + i], (i == NUM_AMD_SMI_FAN_METRICS - 1) ? "" : ", ");
        }
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_FAN_METRICS;

        // PCIe Static
        fprintf(stdout, "PCIeStatic[0..%d]: [", NUM_AMD_SMI_PCIE_STATIC_METRICS - 1);
        for (int i = 0; i < NUM_AMD_SMI_PCIE_STATIC_METRICS; ++i) {
            fprintf(stdout, "%s=%lld%s", amd_smi_pcie_static_names[i], csv_values[print_base_idx + i], (i == NUM_AMD_SMI_PCIE_STATIC_METRICS - 1) ? "" : ", ");
        }
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_PCIE_STATIC_METRICS;

        // Dynamic PCIe
        fprintf(stdout, "PCIeMetric[0..%d]: [", NUM_AMD_SMI_PCIE_METRIC_METRICS - 1);
        for (int i = 0; i < NUM_AMD_SMI_PCIE_METRIC_METRICS; ++i) {
            fprintf(stdout, "%s=%lld%s", amd_smi_pcie_metric_names[i], csv_values[print_base_idx + i], (i == NUM_AMD_SMI_PCIE_METRIC_METRICS - 1) ? "" : ", ");
        }
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_PCIE_METRIC_METRICS;

        // GPU Metrics (Throttle/Residency)
        fprintf(stdout, "GpuMetrics[0..%d]: [", NUM_AMD_SMI_GPU_METRICS - 1);
        fprintf(stdout, "%s=0x%llX", amd_smi_gpu_metric_names[0], csv_values[print_base_idx + 0]);
        fprintf(stdout, ", %s=0x%llX", amd_smi_gpu_metric_names[1], csv_values[print_base_idx + 1]);
        for (int i = 2; i < NUM_AMD_SMI_GPU_METRICS; ++i) {
            fprintf(stdout, ", %s=%lld", amd_smi_gpu_metric_names[i], csv_values[print_base_idx + i]);
        }
        fprintf(stdout, "] | ");
        print_base_idx += NUM_AMD_SMI_GPU_METRICS;


        // Print Process List Info
        fprintf(stdout, "Processes: %u ", num_processes);
        if (smi_status == AMDSMI_STATUS_SUCCESS && num_processes > 0 && proc_list_buffer != NULL) {
            fprintf(stdout, "[");
            for (uint32_t i = 0; i < num_processes; ++i) {
                fprintf(stdout, "{PID: %u, Name: \"%.*s\", VRAM: %llu MiB}%s", // Use %llu for uint64_t
                        proc_list_buffer[i].pid,
                        AMDSMI_MAX_STRING_LENGTH, // Prevent buffer overflow printing name
                        proc_list_buffer[i].name,
                        (unsigned long long)proc_list_buffer[i].mem / (1024 * 1024), // Show VRAM in MiB
                        (i == num_processes - 1) ? "" : ", "); // Separator
            }
            fprintf(stdout, "]");
        } else if (smi_status == AMDSMI_STATUS_NOT_SUPPORTED) {
             fprintf(stdout, "(Not Supported)");
        } else if (smi_status != AMDSMI_STATUS_SUCCESS && params->gpu_handle != NULL) {
             fprintf(stdout, "(Error: %d)", smi_status);
        } else if (params->gpu_handle == NULL) {
            fprintf(stdout, "(No Handle)");
        }
        // If smi_status was SUCCESS but num_processes is 0, it just prints "Processes: 0"

        fprintf(stdout, "\n"); // End of line for stdout

        usleep(params->monitor_interval_us); // Use parameter for sleep duration
    }

    // Cleanup allocated process list buffer
    if (proc_list_buffer) {
        free(proc_list_buffer);
        proc_list_buffer = NULL;
        allocated_proc_list_size = 0;
    }
    fprintf(stdout, "Monitor thread exiting.\n");
    return NULL;
}


// Custom DGEMM kernel (unchanged)
__global__ void dgemm_kernel(const double *A, const double *B, double *C,
                             int M, int N, int K, double alpha, double beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Helper function for cleanup (unchanged)
void cleanup_resources(FILE *csvFile,
                       hipStream_t *streams, hipEvent_t *events, int num_streams,
                       double **d_arrays, int num_d_arrays, // Pass pointers to device arrays
                       double **h_arrays, int num_h_arrays, // Pass pointers to host arrays
                       amdsmi_processor_handle *processor_handles, // Pass pointer to processor handles array
                       amdsmi_processor_handle target_gpu_handle // Pass the target handle itself
                       )
{
    hipError_t hipStatus;
    amdsmi_status_t smi_status;

    printf("Cleaning up resources...\n");

    if (csvFile) {
        fclose(csvFile);
        printf("Closed CSV file.\n");
    }

    // HIP cleanup
    if (streams && events) {
        for (int s = 0; s < num_streams; s++) {
            if (events[s]) {
                hipStatus = hipEventDestroy(events[s]);
                if (hipStatus != hipSuccess) { fprintf(stderr, "Warning: hipEventDestroy failed for event %d: %s\n", s, hipGetErrorString(hipStatus)); }
            }
            if (streams[s]) {
                hipStatus = hipStreamDestroy(streams[s]);
                if (hipStatus != hipSuccess) { fprintf(stderr, "Warning: hipStreamDestroy failed for stream %d: %s\n", s, hipGetErrorString(hipStatus)); }
            }
        }
        printf("HIP streams and events destroyed.\n");
    }

    if (d_arrays) {
        for (int i = 0; i < num_d_arrays; i++) {
            if (d_arrays[i]) {
                hipStatus = hipFree(d_arrays[i]);
                if (hipStatus != hipSuccess) { fprintf(stderr, "Warning: hipFree(d_array[%d]) failed: %s\n", i, hipGetErrorString(hipStatus)); }
            }
        }
         printf("Device memory freed.\n");
    }


    if (h_arrays) {
        for (int i = 0; i < num_h_arrays; i++) {
            if (h_arrays[i]) {
                hipStatus = hipHostFree(h_arrays[i]);
                if (hipStatus != hipSuccess) { fprintf(stderr, "Warning: hipHostFree(h_array[%d]) failed: %s\n", i, hipGetErrorString(hipStatus)); }
            }
        }
        printf("Host memory freed.\n");
    }
    printf("HIP resources freed.\n");


    // AMD SMI Cleanup
    if (target_gpu_handle == NULL && processor_handles != NULL) {
        free(processor_handles);
        printf("Freed processor_handles array (target GPU not found or error before assignment).\n");
    } else if (processor_handles != NULL) {
        // Free the handles array obtained during device discovery
        free(processor_handles);
         printf("Freed processor_handles array.\n");
    }

    // Shut down SMI
    smi_status = amdsmi_shut_down();
    if (smi_status != AMDSMI_STATUS_SUCCESS) { fprintf(stderr, "Warning: AMD SMI shut down failed (Error: %d)\n", smi_status); }
    else { printf("AMD SMI shut down successfully.\n"); }

}


// --- Main function remains largely unchanged ---
int main(int argc, char *argv[]) {
    amdsmi_status_t smi_status;
    hipError_t hipStatus;

    // --- Resource Handles ---
    amdsmi_socket_handle* sockets = NULL;
    amdsmi_processor_handle* processor_handles = NULL; // Array to hold handles from *one* socket during discovery
    amdsmi_processor_handle target_gpu_handle = NULL; // Handle for the specific GPU we use
    double *h_A = NULL, *h_B = NULL, *h_C = NULL; // Host memory
    double *d_A[NUM_STREAMS] = {NULL}, *d_B[NUM_STREAMS] = {NULL}, *d_C[NUM_STREAMS] = {NULL}; // Device memory per stream
    hipStream_t streams[NUM_STREAMS] = {NULL};
    hipEvent_t events[NUM_STREAMS] = {NULL};
    FILE *csvFile = NULL; // Monitoring output file
    pthread_t monitor_thread;
    struct monitor_params params;


    // --- Setup Arrays for Cleanup ---
    double *device_arrays_ptrs[NUM_STREAMS * 3];
    for(int s=0; s<NUM_STREAMS; ++s) {
        device_arrays_ptrs[s*3 + 0] = NULL;
        device_arrays_ptrs[s*3 + 1] = NULL;
        device_arrays_ptrs[s*3 + 2] = NULL;
    }
    double *host_arrays_ptrs[] = {NULL, NULL, NULL}; // Will be updated after allocation


    /* Initialize AMD SMI */
    smi_status = amdsmi_init( AMDSMI_INIT_AMD_GPUS);
     if (smi_status != AMDSMI_STATUS_SUCCESS) {
         fprintf(stderr, "Failed to initialize AMD SMI library (Error: %d)\n", smi_status);
         return -1;
     }
     printf("AMD SMI initialized successfully.\n");

    /* Find the AMD SMI processor handle for the target GPU */
    uint32_t target_logical_gpu_index = 0; // Default to GPU 0
    if (argc > 1) {
        target_logical_gpu_index = atoi(argv[1]);
        printf("Targeting logical GPU index %u from command line argument.\n", target_logical_gpu_index);
    } else {
        printf("Defaulting to logical GPU index %u. You can specify an index as a command line argument.\n", target_logical_gpu_index);
    }

    uint32_t current_gpu_index_counter = 0;
    uint32_t socket_count = 0;
    amdsmi_processor_handle* temp_handles = NULL; // Temporary holder for handles on a specific socket

    smi_status = amdsmi_get_socket_handles(&socket_count, NULL);
    if (smi_status != AMDSMI_STATUS_SUCCESS || socket_count == 0) {
        fprintf(stderr, "Failed to get AMD SMI socket count or no sockets found (Error: %d)\n", smi_status);
        amdsmi_shut_down(); return -1;
    }
    sockets = (amdsmi_socket_handle*)malloc(socket_count * sizeof(amdsmi_socket_handle));
    if (!sockets) { fprintf(stderr, "Failed to allocate memory for sockets.\n"); amdsmi_shut_down(); return -1; }

    smi_status = amdsmi_get_socket_handles(&socket_count, sockets);
    if (smi_status != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to get AMD SMI socket handles (Error: %d)\n", smi_status);
        free(sockets); amdsmi_shut_down(); return -1;
    }

    for (uint32_t i = 0; i < socket_count && target_gpu_handle == NULL; ++i) {
        uint32_t device_count = 0;
        smi_status = amdsmi_get_processor_handles(sockets[i], &device_count, NULL);
        if (smi_status != AMDSMI_STATUS_SUCCESS || device_count == 0) {
            continue; // Try next socket
        }

        temp_handles = (amdsmi_processor_handle*)malloc(device_count * sizeof(amdsmi_processor_handle));
        if (!temp_handles) {
            fprintf(stderr, "Failed to allocate memory for processor handles on socket %u.\n", i);
            if(processor_handles) free(processor_handles); // Free if allocated on previous socket iteration
            free(sockets);
            amdsmi_shut_down(); return -1;
        }

        smi_status = amdsmi_get_processor_handles(sockets[i], &device_count, temp_handles);
        if (smi_status != AMDSMI_STATUS_SUCCESS) {
            fprintf(stderr, "Warning: Failed to get processor handles for socket %u (Error: %d)\n", i, smi_status);
            free(temp_handles); temp_handles = NULL;
            continue;
        }

        for (uint32_t j = 0; j < device_count; ++j) {
            processor_type_t processor_type;
            smi_status = amdsmi_get_processor_type(temp_handles[j], &processor_type);
            if (smi_status == AMDSMI_STATUS_SUCCESS && processor_type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
                if (current_gpu_index_counter == target_logical_gpu_index) {
                    target_gpu_handle = temp_handles[j];
                    printf("Found target GPU handle for logical index %u (Socket %u, Device %u).\n", target_logical_gpu_index, i, j);
                    processor_handles = temp_handles; // Store pointer to free later
                    break; // Found target
                }
                current_gpu_index_counter++;
            }
        }

        if (target_gpu_handle == NULL) {
            free(temp_handles); // Free if target not found on this socket
            temp_handles = NULL;
        }
    }
    free(sockets); sockets = NULL; // Sockets array no longer needed

    if (target_gpu_handle == NULL) {
        fprintf(stderr, "Error: Could not find AMD SMI processor handle for logical GPU index %u.\n", target_logical_gpu_index);
        cleanup_resources(NULL, NULL, NULL, 0, NULL, 0, NULL, 0, processor_handles, NULL);
        return -1;
    }

    /* Set HIP device */
    //(int)target_logical_gpu_index
    int hip_device_index = 1;
    printf("Attempting to set HIP device to index %d (assuming it corresponds to the targeted SMI device).\n", hip_device_index);
    hipStatus = hipSetDevice(hip_device_index);
     if (hipStatus != hipSuccess) {
         fprintf(stderr, "hipSetDevice(%d) failed: %s.\n", hip_device_index, hipGetErrorString(hipStatus));
         fprintf(stderr, "NOTE: HIP device index may not match the logical SMI index used (%u).\n", target_logical_gpu_index);
         fprintf(stderr, "Consider using PCI BDF matching if targeting a specific GPU in a multi-GPU system.\n");
         cleanup_resources(NULL, NULL, NULL, 0, NULL, 0, NULL, 0, processor_handles, target_gpu_handle);
         return -1;
     }
    hipDeviceProp_t deviceProp;
    hipStatus = hipGetDeviceProperties(&deviceProp, hip_device_index);
    if (hipStatus != hipSuccess) {
         fprintf(stderr, "Warning: hipGetDeviceProperties failed for device %d: %s\n", hip_device_index, hipGetErrorString(hipStatus));
    } else {
        printf("HIP Using Device %d: %s\n", hip_device_index, deviceProp.name);
        printf("Compute Units: %d, Max Threads Per Block: %d\n", deviceProp.multiProcessorCount, deviceProp.maxThreadsPerBlock);
    }


    /* Allocate host and device memory, create streams/events */
    size_t size_A = ((size_t)M_DIM * K_DIM * sizeof(double));
    size_t size_B = ((size_t)K_DIM * N_DIM * sizeof(double));
    size_t size_C = ((size_t)M_DIM * N_DIM * sizeof(double));

    hipStatus = hipHostMalloc(&h_A, size_A, hipHostMallocDefault);
      if (hipStatus != hipSuccess) { fprintf(stderr, "hipHostMalloc h_A failed: %s\n", hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, NUM_STREAMS, NULL, 0, NULL, 0, processor_handles, target_gpu_handle); return -1; }
      host_arrays_ptrs[0] = h_A;
    hipStatus = hipHostMalloc(&h_B, size_B, hipHostMallocDefault);
      if (hipStatus != hipSuccess) { fprintf(stderr, "hipHostMalloc h_B failed: %s\n", hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, NUM_STREAMS, NULL, 0, host_arrays_ptrs, 1, processor_handles, target_gpu_handle); return -1; }
      host_arrays_ptrs[1] = h_B;
    hipStatus = hipHostMalloc(&h_C, size_C, hipHostMallocDefault);
      if (hipStatus != hipSuccess) { fprintf(stderr, "hipHostMalloc h_C failed: %s\n", hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, NUM_STREAMS, NULL, 0, host_arrays_ptrs, 2, processor_handles, target_gpu_handle); return -1; }
      host_arrays_ptrs[2] = h_C;
      printf("Host memory allocated.\n");

    for (size_t i = 0; i < (size_t)M_DIM * K_DIM; i++) h_A[i] = (double)(i % 100);
    for (size_t i = 0; i < (size_t)K_DIM * N_DIM; i++) h_B[i] = (double)(i % 100);
    for (size_t i = 0; i < (size_t)M_DIM * N_DIM; i++) h_C[i] = 0.0;
    printf("Host data initialized.\n");

    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStatus = hipMalloc((void**)&d_A[s], size_A);
        if (hipStatus != hipSuccess) { fprintf(stderr, "hipMalloc d_A[%d] failed: %s\n", s, hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, s, device_arrays_ptrs, s*3+0, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1; }
        device_arrays_ptrs[s*3 + 0] = d_A[s];
        hipStatus = hipMalloc((void**)&d_B[s], size_B);
        if (hipStatus != hipSuccess) { fprintf(stderr, "hipMalloc d_B[%d] failed: %s\n", s, hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, s, device_arrays_ptrs, s*3+1, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1; }
        device_arrays_ptrs[s*3 + 1] = d_B[s];
        hipStatus = hipMalloc((void**)&d_C[s], size_C);
        if (hipStatus != hipSuccess) { fprintf(stderr, "hipMalloc d_C[%d] failed: %s\n", s, hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, s, device_arrays_ptrs, s*3+2, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1; }
        device_arrays_ptrs[s*3 + 2] = d_C[s];

        hipStatus = hipStreamCreateWithFlags(&streams[s], hipStreamNonBlocking);
        if (hipStatus != hipSuccess) { fprintf(stderr, "hipStreamCreate failed stream %d: %s\n", s, hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, s+1, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1; }
        hipStatus = hipEventCreate(&events[s]);
        if (hipStatus != hipSuccess) { fprintf(stderr, "hipEventCreate failed event %d: %s\n", s, hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, s+1, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1; }
    }
    printf("Device memory allocated.\n");
    printf("HIP streams and events created.\n");


    // Initial H2D Copy
    printf("Starting initial H2D memory copies...\n");
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStatus = hipMemcpyAsync(d_A[s], h_A, size_A, hipMemcpyHostToDevice, streams[s]);
        if (hipStatus != hipSuccess) { fprintf(stderr, "hipMemcpyAsync d_A[%d] failed: %s\n", s, hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, NUM_STREAMS, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1; }
        hipStatus = hipMemcpyAsync(d_B[s], h_B, size_B, hipMemcpyHostToDevice, streams[s]);
        if (hipStatus != hipSuccess) { fprintf(stderr, "hipMemcpyAsync d_B[%d] failed: %s\n", s, hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, NUM_STREAMS, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1; }
        hipStatus = hipMemcpyAsync(d_C[s], h_C, size_C, hipMemcpyHostToDevice, streams[s]);
        if (hipStatus != hipSuccess) { fprintf(stderr, "hipMemcpyAsync d_C[%d] failed: %s\n", s, hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, NUM_STREAMS, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1; }
    }
    for (int s = 0; s < NUM_STREAMS; s++) { // Ensure copies complete
        hipStatus = hipStreamSynchronize(streams[s]);
        if (hipStatus != hipSuccess) { fprintf(stderr, "hipStreamSynchronize H2D failed stream %d: %s\n", s, hipGetErrorString(hipStatus)); cleanup_resources(NULL, streams, events, NUM_STREAMS, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1; }
    }
    printf("Initial H2D memory copies completed.\n");


    /* Open CSV file and write header. */
    csvFile = fopen("gemm_monitoring_amdsmi_gpu_metrics_refactored.csv", "w"); // Changed filename slightly
    if (!csvFile) {
        fprintf(stderr, "Failed to open CSV file for writing.\n");
        cleanup_resources(NULL, streams, events, NUM_STREAMS, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle); return -1;
    }
    // Write header for metrics being collected into the CSV (order matches TOTAL_CSV_METRICS definition)
    fprintf(csvFile, "timestamp");
    for(int i = 0; i < NUM_AMD_SMI_SENSORS; ++i) fprintf(csvFile, ",%s", amd_smi_sensors[i].name);
    for(int i = 0; i < NUM_AMD_SMI_ACTIVITY_METRICS; ++i) fprintf(csvFile, ",%s", amd_smi_activity_names[i]);
    for(int i = 0; i < NUM_AMD_SMI_VRAM_METRICS; ++i) fprintf(csvFile, ",%s", amd_smi_vram_names[i]);
    for(int i = 0; i < NUM_AMD_SMI_POWER_INFO_METRICS; ++i) fprintf(csvFile, ",%s", amd_smi_power_info_names[i]);
    for(int i = 0; i < NUM_AMD_SMI_CLOCK_METRICS; ++i) fprintf(csvFile, ",%s", amd_smi_clocks[i].name);
    for(int i = 0; i < NUM_AMD_SMI_CLOCK_INFO_METRICS; ++i) fprintf(csvFile, ",%s", amd_smi_clock_info_defs[i].name);
    for(int i = 0; i < NUM_AMD_SMI_FAN_METRICS; ++i) fprintf(csvFile, ",%s", amd_smi_fan_speed_names[i]);
    for(int i = 0; i < NUM_AMD_SMI_PCIE_STATIC_METRICS; ++i) fprintf(csvFile, ",%s", amd_smi_pcie_static_names[i]);
    for(int i = 0; i < NUM_AMD_SMI_PCIE_METRIC_METRICS; ++i) fprintf(csvFile, ",%s", amd_smi_pcie_metric_names[i]);
    for(int i = 0; i < NUM_AMD_SMI_GPU_METRICS; ++i) fprintf(csvFile, ",%s", amd_smi_gpu_metric_names[i]);
    fprintf(csvFile, ",NumProcesses"); // Add header for the process count
    fprintf(csvFile, "\n"); // End header row
    fflush(csvFile);
    printf("CSV file opened and header written.\n");


    /* Start the monitoring thread */
    params.gpu_handle = target_gpu_handle;
    params.csvFile = csvFile;
    params.monitor_interval_us = 100000; // Set interval (0.3 seconds)
    gettimeofday(&params.start_time, NULL); // Get start time just before thread creation

    stop_monitor = 0; // Ensure flag is reset
    int thread_status = pthread_create(&monitor_thread, NULL, monitor_events, &params);
    if (thread_status != 0) {
        fprintf(stderr, "pthread_create failed with error code %d.\n", thread_status);
        cleanup_resources(csvFile, streams, events, NUM_STREAMS, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle);
        return -1;
    }
    printf("Monitoring thread started.\n");

    /* GEMM kernel launch loop */
    double alpha = 0.75; double beta  = 0.5;
    dim3 blockDim(32, 32);
    dim3 gridDim((N_DIM + blockDim.x - 1) / blockDim.x,
                 (M_DIM + blockDim.y - 1) / blockDim.y);

    printf("Launching DGEMM kernel (%d iterations per stream)...\n", ITERATIONS_PER_STREAM);
    usleep(5000000); // Small delay before starting work

    for (int iter = 0; iter < ITERATIONS_PER_STREAM; iter++) {
        for (int s = 0; s < NUM_STREAMS; s++) {
            hipLaunchKernelGGL(dgemm_kernel, gridDim, blockDim, 0, streams[s],
                               d_A[s], d_B[s], d_C[s],
                               M_DIM, N_DIM, K_DIM, alpha, beta);
             hipStatus = hipGetLastError();
             if (hipStatus != hipSuccess) {
                 fprintf(stderr, "Kernel launch failed iter %d stream %d: %s\n", iter, s, hipGetErrorString(hipStatus));
                 stop_monitor = 1; // Signal monitor to stop
                 pthread_join(monitor_thread, NULL); // Wait for monitor
                 cleanup_resources(csvFile, streams, events, NUM_STREAMS, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle);
                 return -1; // Exit after cleanup
             }

            hipStatus = hipEventRecord(events[s], streams[s]);
             if (hipStatus != hipSuccess) { fprintf(stderr, "Warning: hipEventRecord failed iter %d stream %d: %s\n", iter, s, hipGetErrorString(hipStatus));}

        }
        // Wait for all streams in this iteration to finish
        for (int s = 0; s < NUM_STREAMS; s++) {
            hipStatus = hipStreamSynchronize(streams[s]);
            if (hipStatus != hipSuccess) {
                fprintf(stderr, "Stream sync failed iter %d stream %d: %s\n", iter, s, hipGetErrorString(hipStatus));
                stop_monitor = 1; pthread_join(monitor_thread, NULL);
                cleanup_resources(csvFile, streams, events, NUM_STREAMS, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle);
                return -1;
            }
        }
         printf("Completed iteration %d\n", iter + 1);
         usleep(5000000); // 5 sec delay between iterations
    }
    printf("DGEMM kernel executions finished.\n");


    /* Copy results back D2H (optional, from stream 0) */
    printf("Copying results D2H (stream 0)...\n");
    hipStatus = hipMemcpyAsync(h_C, d_C[0], size_C, hipMemcpyDeviceToHost, streams[0]);
    if (hipStatus != hipSuccess) { fprintf(stderr, "hipMemcpyAsync D2H failed: %s\n", hipGetErrorString(hipStatus)); }
    else {
        hipStatus = hipStreamSynchronize(streams[0]); // Wait for copy to finish
        if (hipStatus != hipSuccess) { fprintf(stderr, "Stream sync after D2H failed: %s\n", hipGetErrorString(hipStatus)); }
        else { printf("Results copied back successfully.\n"); }
    }


    /* Stop monitor thread */
    printf("Stopping monitor thread...\n");
    stop_monitor = 1;
    pthread_join(monitor_thread, NULL);
    printf("Monitor thread joined.\n");


    /* Cleanup resources */
    cleanup_resources(csvFile, streams, events, NUM_STREAMS, device_arrays_ptrs, NUM_STREAMS*3, host_arrays_ptrs, 3, processor_handles, target_gpu_handle);


    printf("Execution finished.\n");
    return 0;
}