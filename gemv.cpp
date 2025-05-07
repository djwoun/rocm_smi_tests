#include <stdio.h>
#include <stdlib.h>
#include "papi.h"
#include <hip/hip_runtime.h>
#include <unistd.h>      // For usleep()
#include <sys/time.h>    // For gettimeofday()
#include <pthread.h>     // For pthreads

// Define dimensions for DGEMV
#define M_DIM 58368  // Number of rows in matrix A and length of result vector y
#define K_DIM 116736   // Number of columns in matrix A and length of vector x

// Number of streams to use for concurrent execution
#define NUM_STREAMS 1

// Number of iterations to run in each stream
#define ITERATIONS_PER_STREAM 20

// Global flag to signal the monitor thread to stop.
volatile int stop_monitor = 0;

// Structure to pass parameters to the monitoring thread.
struct monitor_params {
    int EventSet;
    FILE *csvFile;
    struct timeval start_time;
};

// Monitor thread that periodically reads PAPI counters and logs them.
void *monitor_events(void *args) {
    struct monitor_params *params = (struct monitor_params *)args;
    int statusFlag;
    long long values[6];  // now monitoring 6 events (event0 through event5)

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
        fprintf(params->csvFile, "%.6f,%lld,%lld,%lld,%lld,%lld,%lld\n",
                elapsed, values[0], values[1], values[2], values[3], values[4], values[5]);
        fflush(params->csvFile);

        // Also print to stdout, including event0.
        fprintf(stdout,
                "Time: %.6f sec -> event0: %lld, "
                "rocm_smi:::temp_current:device=0:sensor=1: %lld, "
                "rocm_smi:::temp_current:device=0:sensor=2: %lld, "
                "rocm_smi:::mem_usage_VRAM:device=0: %lld, "
                "rocm_smi:::busy_percent:device=0: %lld, "
                "rocm_smi:::memory_busy_percent:device=0: %lld\n",
                elapsed, values[0], values[1], values[2], values[3], values[4], values[5]);

        usleep(100000);  // Sleep for X seconds.
    }
    return NULL;
}

// Custom DGEMV kernel using a simple row-major implementation.
// Computes: y[row] = alpha * dot(A[row][:], x) + beta * y[row]
__global__ void dgemv_kernel(const double *A, const double *x, double *y,
                             size_t M, size_t N, double alpha, double beta) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double sum = 0.0;
        for (size_t j = 0; j < N; j++) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = alpha * sum + beta * y[row];
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

    /* Add GPU events to the event set (now including event0). */
    const char *event0 = "rocm_smi:::gpu_clk_freq_System:device=0:idx=0";
    const char *event1 = "rocm_smi:::temp_current:device=0:sensor=1";
    const char *event2 = "rocm_smi:::temp_current:device=0:sensor=2";
    const char *event3 = "rocm_smi:::mem_usage_VRAM:device=0";
    const char *event4 = "rocm_smi:::busy_percent:device=0";
    const char *event5 = "rocm_smi:::memory_busy_percent:device=0";

    statusFlag = PAPI_add_named_event(EventSet, event0);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI add named event 0: %s\n", PAPI_strerror(statusFlag));
        return -1;
    }
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

    /* Set HIP device properties to optimize for MI300 */
    hipSetDevice(1);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, 1);
    printf("Device Name: %s\n", deviceProp.name);
    printf("Compute Units: %d\n", deviceProp.multiProcessorCount);
    printf("Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
    
    /* Allocate host memory for matrix A, vector x, and vector y with page-locked memory for faster transfers */
    size_t size_A = ((size_t)M_DIM * K_DIM * sizeof(double));
    size_t size_B = ((size_t)K_DIM * sizeof(double));  // Vector x
    size_t size_C = ((size_t)M_DIM * sizeof(double));   // Vector y

    double *h_A, *h_B, *h_C;
    hipHostMalloc(&h_A, size_A, hipHostMallocDefault);
    hipHostMalloc(&h_B, size_B, hipHostMallocDefault);
    hipHostMalloc(&h_C, size_C, hipHostMallocDefault);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return -1;
    }

    /* Initialize host matrix A and vectors x and y. */
    for (int i = 0; i < M_DIM * K_DIM; i++) {
        h_A[i] = (double)(i % 100);
    }
    for (int i = 0; i < K_DIM; i++) {
        h_B[i] = (double)(i % 100);
    }
    for (int i = 0; i < M_DIM; i++) {
        h_C[i] = 0.0;
    }

    /* Allocate device memory. */
    double *d_A[NUM_STREAMS], *d_B[NUM_STREAMS], *d_C[NUM_STREAMS];
    hipError_t hipStatus;
    
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStatus = hipMalloc((void**)&d_A[s], size_A);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMalloc d_A[%d] failed.\n", s);
            return -1;
        }
        hipStatus = hipMalloc((void**)&d_B[s], size_B);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMalloc d_B[%d] failed.\n", s);
            return -1;
        }
        hipStatus = hipMalloc((void**)&d_C[s], size_C);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMalloc d_C[%d] failed.\n", s);
            return -1;
        }
    }

    /* Create multiple streams for concurrent execution */
    hipStream_t streams[NUM_STREAMS];
    hipEvent_t events[NUM_STREAMS];
    
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStatus = hipStreamCreateWithFlags(&streams[s], hipStreamNonBlocking);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipStreamCreate failed for stream %d.\n", s);
            return -1;
        }
        
        hipStatus = hipEventCreate(&events[s]);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipEventCreate failed for event %d.\n", s);
            return -1;
        }
    }

    /* Copy host data to device memory in parallel across streams */
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStatus = hipMemcpyAsync(d_A[s], h_A, size_A, hipMemcpyHostToDevice, streams[s]);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMemcpyAsync d_A[%d] failed.\n", s);
            return -1;
        }
        hipStatus = hipMemcpyAsync(d_B[s], h_B, size_B, hipMemcpyHostToDevice, streams[s]);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMemcpyAsync d_B[%d] failed.\n", s);
            return -1;
        }
        hipStatus = hipMemcpyAsync(d_C[s], h_C, size_C, hipMemcpyHostToDevice, streams[s]);
        if (hipStatus != hipSuccess) {
            fprintf(stderr, "hipMemcpyAsync d_C[%d] failed.\n", s);
            return -1;
        }
    }

    /* Open CSV file for recording data and write header. */
    FILE *csvFile = fopen("test.csv", "w");
    if (!csvFile) {
        fprintf(stderr, "Failed to open CSV file for writing.\n");
        return -1;
    }
    fprintf(csvFile, "timestamp,%s,%s,%s,%s,%s,%s\n",
            event0, event1, event2, event3, event4, event5);

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

    /* Wait for the initial copies to complete */
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipStreamSynchronize(streams[s]);
    }

    /* DGEMV parameters */
    double alpha = 0.75;
    double beta  = 0.5;
    
    // Define grid and block dimensions for the GEMV kernel launch.
    // Here we use a 1D grid as the kernel computes a vector of length M_DIM.
    int threadsPerBlock = 128;
    int blocksPerGrid = (M_DIM + threadsPerBlock - 1) / threadsPerBlock;
    
    usleep(5000000);
     
    /* Kernel execution loop to keep the GPU busy */
    for (int iter = 0; iter < ITERATIONS_PER_STREAM; iter++) {
        for (int s = 0; s < NUM_STREAMS; s++) {
            // Launch the custom DGEMV kernel on stream 's'
            hipLaunchKernelGGL(dgemv_kernel, blocksPerGrid, threadsPerBlock, 0, streams[s],
                               d_A[s], d_B[s], d_C[s],
                               M_DIM, K_DIM, alpha, beta);

            // Record event but don't synchronize.
            hipEventRecord(events[s], streams[s]);
            
            hipStreamSynchronize(streams[s]);
            
            usleep(5000000);
        }
    }
    
    /* Copy results back from one stream as we only need one copy */
    hipStatus = hipMemcpyAsync(h_C, d_C[0], size_C, hipMemcpyDeviceToHost, streams[0]);
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMemcpy h_C failed.\n");
        return -1;
    }
    
    /* Wait for all streams to complete */
    //for (int s = 0; s < NUM_STREAMS; s++) {
    //    hipStreamSynchronize(streams[s]);
    //}
    hipStreamSynchronize(streams[0]);

    /* Signal the monitor thread to stop and wait for it to finish. */
    stop_monitor = 1;
    pthread_join(monitor_thread, NULL);

    /* Cleanup resources. */
    fclose(csvFile);
    
    for (int s = 0; s < NUM_STREAMS; s++) {
        hipEventDestroy(events[s]);
        hipStreamDestroy(streams[s]);
        hipFree(d_A[s]);
        hipFree(d_B[s]);
        hipFree(d_C[s]);
    }
    
    hipHostFree(h_A);
    hipHostFree(h_B);
    hipHostFree(h_C);

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
