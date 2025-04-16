#include <stdio.h>
#include <stdlib.h>
#include <papi.h>
#include <hip/hip_runtime.h>
#include <unistd.h>      // For usleep()
#include <sys/time.h>    // For gettimeofday()
#include <math.h>        // For sin() and cos()

//#define N 2048*2048  // Increase N for more compute work
#define N (32768*32768)

// A heavy HIP kernel that performs a computationally intensive task.
__global__ void heavyVectorAdd(double *a, double *b, double *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double value = a[i] + b[i];
        // Perform heavy computation: 10,000 iterations of trigonometric operations.
        for (int j = 0; j < 10000; j++) {
            value = sin(value) + cos(value);
        }
        c[i] = value;
    }
}

int main(int argc, char *argv[]) {
    int statusFlag;
    int EventSet = PAPI_NULL;

    /* Initialize PAPI. */
    statusFlag = PAPI_library_init(PAPI_VER_CURRENT);
    if (statusFlag != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI shared library version error: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Create event set. */
    statusFlag = PAPI_create_eventset(&EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Add GPU events to the event set. */
    const char *event1 = "rocm_smi:::temp_current:device=1:sensor=1";
    const char *event2 = "rocm_smi:::temp_current:device=1:sensor=2";
    const char *event3 = "rocm_smi:::busy_percent:device=1";
    const char *event4 = "rocm_smi:::memory_busy_percent:device=1";
    
    statusFlag = PAPI_add_named_event(EventSet, event1);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 1: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    statusFlag = PAPI_add_named_event(EventSet, event2);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 2: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    statusFlag = PAPI_add_named_event(EventSet, event3);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 3: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    statusFlag = PAPI_add_named_event(EventSet, event4);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 4: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Allocate host memory for vectors. */
    size_t size = N * sizeof(double);
    double *h_a = (double*)malloc(size);
    double *h_b = (double*)malloc(size);
    double *h_c = (double*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return -1;
    }

    // Initialize input data.
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0;
        h_b[i] = i * 2.0;
    }

    hipSetDevice(1);
    
    /* Allocate device memory. */
    double *d_a, *d_b, *d_c;
    hipError_t hipStatus;
    hipStatus = hipMalloc((void**)&d_a, size);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMalloc d_a failed.\n");
        return -1;
    }
    hipStatus = hipMalloc((void**)&d_b, size);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMalloc d_b failed.\n");
        return -1;
    }
    hipStatus = hipMalloc((void**)&d_c, size);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMalloc d_c failed.\n");
        return -1;
    }

    /* Copy input data from host to device. */
    hipStatus = hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy d_a failed.\n");
        return -1;
    }
    hipStatus = hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy d_b failed.\n");
        return -1;
    }

    /* Open CSV file for recording sensor data. */
    FILE *csvFile = fopen("papi_measurements3.csv", "w");
    if (!csvFile) {
        fprintf(stderr, "Failed to open CSV file for writing.\n");
        return -1;
    }
    // Write CSV header: timestamp and the four events.
    fprintf(csvFile, "timestamp,%s,%s,%s,%s,%s\n", event1, event2, event3, event4, "power");

    /* Start PAPI counters to monitor GPU metrics. */
    statusFlag = PAPI_start(EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI start: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Record the start time. */
    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    /* Create a HIP event to detect when the kernel finishes. */
    hipEvent_t kernelDone;
    hipStatus = hipEventCreate(&kernelDone);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipEventCreate failed.\n");
        return -1;
    }

    /* Launch GPU kernel asynchronously. */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(heavyVectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, N);
    // Record the event immediately after kernel launch.
    hipEventRecord(kernelDone, 0);

    /* Continuously record sensor readings every 0.5 second until the kernel is done. */
    long long values[4];
    while (hipEventQuery(kernelDone) == hipErrorNotReady) {
        statusFlag = PAPI_read(EventSet, values);
        if (statusFlag != PAPI_OK) {
            fprintf(stderr, "PAPI read: %s\n", PAPI_strerror(statusFlag));
            break;
        }
        // Get current time and compute elapsed time in seconds.
        struct timeval current_time;
        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - start_time.tv_sec) +
                         (current_time.tv_usec - start_time.tv_usec) / 1e6;
                         
                         
        int gpu1_power = -1; // Default to -1 (error/unavailable)
        FILE *fp = popen("amd-smi metric -g 1 -p --csv", "r"); // Use specific command
        if (fp != NULL) {
            char buffer[128]; // Sufficient buffer for the expected output
            int header_skipped = 0;
            int data_parsed = 0;

            while (fgets(buffer, sizeof(buffer), fp) != NULL) {
                // Skip the header line (contains "gpu")
                if (!header_skipped && strstr(buffer, "gpu")) {
                    header_skipped = 1;
                    continue;
                }
                // Parse the data line (after header)
                if (header_skipped) {
                    int gpu_id_read;
                    // Expect format like "1,83,..." - parse first two ints
                    if (sscanf(buffer, "%d,%d", &gpu_id_read, &gpu1_power) == 2) {
                        data_parsed = 1; // Flag success
                        break;          // Got the data, no need to read further
                    } else {
                        // Failed to parse data line, treat as error for this sample
                        gpu1_power = -1;
                        break;
                    }
                }
            }

            // Check if data was actually parsed after skipping header
            if (header_skipped && !data_parsed) {
                gpu1_power = -1; // Header found, but data parsing failed/missing
            } else if (!header_skipped) {
                 gpu1_power = -1; // Header wasn't even found
            }


            int status = pclose(fp);
            // If command failed execution, ensure power is marked as error
            if (status == -1 || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) {
                 if (gpu1_power != -1) { // Only print warning if we previously thought we succeeded
                      // Optional: fprintf(stderr, "Warning: amd-smi command failed, but power value was parsed earlier.\n");
                 }
                 gpu1_power = -1;
            }
        } else {
             perror("Failed to run amd-smi"); // popen failed itself
             // gpu1_power remains -1
        }

        // Write measurements to the CSV file.
        fprintf(csvFile, "%.6f,%lld,%lld,%lld,%lld,%lld\n", elapsed, values[0], values[1], values[2], values[3],gpu1_power);

        // Optionally print to stdout.
        fprintf(stdout, "Time: %.6f sec -> %s: %lld, %s: %lld, %s: %lld, %s: %lld %s: %lld\n",
                elapsed, event1, values[0], event2, values[1], event3, values[2], event4, values[3], "power" , gpu1_power);
        fflush(stdout);
        usleep(300000);  // Sleep for 500 milliseconds.
    }

    /* Wait for the kernel to complete. */
    hipEventSynchronize(kernelDone);

    /* Copy result back from device to host (if needed). */
    hipStatus = hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy h_c failed.\n");
        return -1;
    }

    /* Cleanup resources. */
    fclose(csvFile);
    hipEventDestroy(kernelDone);
    free(h_a);
    free(h_b);
    free(h_c);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    statusFlag = PAPI_stop(EventSet, values);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI stop: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    
    statusFlag = PAPI_cleanup_eventset(EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI cleanup eventset: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
    statusFlag = PAPI_destroy_eventset(&EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI destroy eventset: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    return 0;
}
