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
    // Write CSV header: timestamp, iteration info and the four events.
    fprintf(csvFile, "timestamp,iteration,%s,%s,%s,%s\n", event1, event2, event3, event4);

    /* Start PAPI counters to monitor GPU metrics. */
    statusFlag = PAPI_start(EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI start: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Record the global start time. */
    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    /* Create a HIP event to detect when the kernel finishes. */
    hipEvent_t kernelDone;
    hipStatus = hipEventCreate(&kernelDone);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipEventCreate failed.\n");
        return -1;
    }

    /* Determine kernel launch configuration. */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Run the kernel 20 times with continuous monitoring. */
    for (int iter = 0; iter < 20; iter++) {
        // Launch GPU kernel asynchronously.
        hipLaunchKernelGGL(heavyVectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, N);
        // Record the event immediately after kernel launch.
        hipEventRecord(kernelDone, 0);
        
        /* Continuously record sensor readings every 0.5 second while the kernel is running. */
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

            // Write measurements to the CSV file with iteration info.
            fprintf(csvFile, "%.6f,Iter%d,%lld,%lld,%lld,%lld\n", elapsed, iter + 1, values[0], values[1], values[2], values[3]);
            
            // Optionally print to stdout.
            fprintf(stdout, "Iteration %d, Time: %.6f sec -> %s: %lld, %s: %lld, %s: %lld, %s: %lld\n",
                    iter + 1, elapsed, event1, values[0], event2, values[1], event3, values[2], event4, values[3]);
            fflush(stdout);
            usleep(500000);  // Sleep for 500 milliseconds.
        }
        
        /* Wait for the kernel to complete. */
        hipEventSynchronize(kernelDone);
        
        /* After the kernel finishes, continuously record sensor readings for an additional 5 seconds. */
        struct timeval wait_start, wait_current;
        gettimeofday(&wait_start, NULL);
        double wait_elapsed = 0.0;
        while (wait_elapsed < 5.0) {
            statusFlag = PAPI_read(EventSet, values);
            if (statusFlag != PAPI_OK) {
                fprintf(stderr, "PAPI read during wait: %s\n", PAPI_strerror(statusFlag));
                break;
            }
            gettimeofday(&wait_current, NULL);
            wait_elapsed = (wait_current.tv_sec - wait_start.tv_sec) +
                           (wait_current.tv_usec - wait_start.tv_usec) / 1e6;
            double overall_elapsed = (wait_current.tv_sec - start_time.tv_sec) +
                                     (wait_current.tv_usec - start_time.tv_usec) / 1e6;
            // Write wait period measurements to the CSV file with iteration wait indicator.
            fprintf(csvFile, "%.6f,Iter%d-Wait,%lld,%lld,%lld,%lld\n", overall_elapsed, iter + 1, values[0], values[1], values[2], values[3]);
            
            fprintf(stdout, "Iteration %d Wait, Time: %.6f sec -> %s: %lld, %s: %lld, %s: %lld, %s: %lld\n",
                    iter + 1, overall_elapsed, event1, values[0], event2, values[1], event3, values[2], event4, values[3]);
            fflush(stdout);
            usleep(500000);  // Sleep for 500 milliseconds.
        }
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

    statusFlag = PAPI_stop(EventSet, NULL);
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
