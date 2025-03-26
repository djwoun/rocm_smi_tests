#include <stdio.h>
#include <stdlib.h>
#include <papi.h>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <unistd.h>      // For usleep()
#include <sys/time.h>    // For gettimeofday()
#include <pthread.h>     // For pthreads

// Define matrix dimensions for GEMM.
#define M_DIM 62768  // Number of rows of A and C
#define K_DIM 62768  // Number of columns of A and rows of B
#define N_DIM 62768  // Number of columns of B and C

// Global flag to signal the monitor thread to stop.
volatile int stop_monitor = 0;

// Structure to pass parameters to the monitoring thread.
struct monitor_params {
    int EventSet;
    FILE *csvFile;
    struct timeval start_time;
    // You can pass the event names if needed.
};

void *monitor_events(void *args) {
    struct monitor_params *params = (struct monitor_params *)args;
    int statusFlag;
    long long values[5];

    // Continue monitoring until stop_monitor is set.
    while (!stop_monitor) {
        statusFlag = PAPI_read(params->EventSet, values);
        if (statusFlag != PAPI_OK) {
            fprintf(stderr, "PAPI read failed in monitor: %s\n", PAPI_strerror(statusFlag));
            break;
        }

        struct timeval current_time;
        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - params->start_time.tv_sec) +
                         (current_time.tv_usec - params->start_time.tv_usec) / 1e6;

        // Write the measurements to the CSV file.
        fprintf(params->csvFile, "%.6f,%lld,%lld,%lld,%lld,%lld\n",
                elapsed, values[0], values[1], values[2], values[3], values[4]);
        fflush(params->csvFile);

        // Also print to stdout.
        fprintf(stdout,
                "Time: %.6f sec -> rocm_smi:::temp_current:device=1:sensor=1: %lld, "
                "rocm_smi:::temp_current:device=1:sensor=2: %lld, "
                "rocm_smi:::mem_usage_VRAM:device=1: %lld, "
                "rocm_smi:::busy_percent:device=1: %lld, "
                "rocm_smi:::memory_busy_percent:device=1: %lld\n",
                elapsed, values[0], values[1], values[2], values[3], values[4]);

        usleep(500000);  // Sleep for 0.5 seconds.
    }

    return NULL;
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
    const char *event3 = "rocm_smi:::mem_usage_VRAM:device=1";
    const char *event4 = "rocm_smi:::busy_percent:device=1";
    const char *event5 = "rocm_smi:::memory_busy_percent:device=1";  // New event

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
    statusFlag = PAPI_add_named_event(EventSet, event5);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 5: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Allocate host memory for matrices A, B, and C. */
    size_t size_A = ((size_t)M_DIM * K_DIM * sizeof(double));
    size_t size_B = ((size_t)K_DIM * N_DIM * sizeof(double));
    size_t size_C = ((size_t)M_DIM * N_DIM * sizeof(double));

    double *h_A = (double*)malloc(size_A);
    double *h_B = (double*)malloc(size_B);
    double *h_C = (double*)malloc(size_C);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return -1;
    }

    /* Initialize host matrices. */
    for (int i = 0; i < M_DIM * K_DIM; i++) {
        h_A[i] = (double)(i % 100);
    }
    for (int i = 0; i < K_DIM * N_DIM; i++) {
        h_B[i] = (double)(i % 100);
    }
    for (int i = 0; i < M_DIM * N_DIM; i++) {
        h_C[i] = 0.0;
    }

    /* Allocate device memory. */
    double *d_A, *d_B, *d_C;
    hipSetDevice(1);

    hipError_t hipStatus;
    hipStatus = hipMalloc((void**)&d_A, size_A);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMalloc d_A failed.\n");
        return -1;
    }
    hipStatus = hipMalloc((void**)&d_B, size_B);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMalloc d_B failed.\n");
        return -1;
    }
    hipStatus = hipMalloc((void**)&d_C, size_C);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMalloc d_C failed.\n");
        return -1;
    }

    /* Copy host matrices to device memory. */
    hipStatus = hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy d_A failed.\n");
        return -1;
    }
    hipStatus = hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy d_B failed.\n");
        return -1;
    }
    hipStatus = hipMemcpy(d_C, h_C, size_C, hipMemcpyHostToDevice);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy d_C failed.\n");
        return -1;
    }

    /* Create rocBLAS handle. */
    rocblas_handle handle;
    rocblas_status rb_status;
    rb_status = rocblas_create_handle(&handle);
    if (rb_status != rocblas_status_success) {
        fprintf(stderr, "rocblas_create_handle failed.\n");
        return -1;
    }

    /* Create a HIP event to detect GEMM completion. */
    hipEvent_t gemmDone;
    hipStatus = hipEventCreate(&gemmDone);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipEventCreate failed.\n");
        return -1;
    }

    /* Open CSV file for recording data and write header. */
    FILE *csvFile = fopen("papi_measurements.csv", "w");
    if (!csvFile) {
        fprintf(stderr, "Failed to open CSV file for writing.\n");
        return -1;
    }
    fprintf(csvFile, "timestamp,%s,%s,%s,%s,%s\n",
            event1, event2, event3, event4, event5);

    /* Start PAPI counters to monitor GPU metrics. */
    statusFlag = PAPI_start(EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI start: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }

    /* Start the monitoring thread immediately after PAPI_start */
    pthread_t monitor_thread;
    struct monitor_params params;
    params.EventSet = EventSet;
    params.csvFile = csvFile;
    gettimeofday(&params.start_time, NULL);  // Record the start time

    statusFlag = pthread_create(&monitor_thread, NULL, monitor_events, &params);
    if (statusFlag != 0) {
        fprintf(stderr, "pthread_create failed.\n");
        return -1;
    }

    /* Launch GEMM operation asynchronously.
       Compute: C = alpha * A * B + beta * C */
    double alpha = 0.75;
    double beta  = 0.5;
    int iterations = 5; // Adjust as needed

    for (int iter = 0; iter < iterations; iter++) {
        rb_status = rocblas_dgemm(handle,
                                  rocblas_operation_none, rocblas_operation_none,
                                  M_DIM, N_DIM, K_DIM,
                                  &alpha,
                                  d_A, M_DIM,
                                  d_B, K_DIM,
                                  &beta,
                                  d_C, M_DIM);
        if (rb_status != rocblas_status_success) {
            fprintf(stderr, "rocblas_dgemm failed on iteration %d.\n", iter);
            return -1;
        }
        // Record the GEMM event.
        hipEventRecord(gemmDone, 0);

        // Instead of a loop here, the monitor thread is already recording every 0.5 sec.
        hipEventSynchronize(gemmDone);
    }

    hipEventSynchronize(gemmDone);

    /* Copy the result from device to host. */
    hipStatus = hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy h_C failed.\n");
        return -1;
    }

    /* Signal the monitor thread to stop and wait for it to finish. */
    stop_monitor = 1;
    pthread_join(monitor_thread, NULL);

    /* Cleanup resources. */
    fclose(csvFile);
    hipEventDestroy(gemmDone);
    rocblas_destroy_handle(handle);
    free(h_A);
    free(h_B);
    free(h_C);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

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
