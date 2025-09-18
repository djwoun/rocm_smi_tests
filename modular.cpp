// gemm.cpp  — unified runner with pluggable monitoring (PAPI | AMD SMI)
// and pluggable compute: custom DGEMM kernel ("cus") or rocBLAS DGEMM ("rocblas")
//
// Usage:
//   ./gemm papi  rocblas [device]   | ./gemm papi  cus [device]
//   ./gemm amd   rocblas [device]   | ./gemm amd   cus [device]
//   ./gemm help
//
// Build:
//   hipcc gemm.cpp -o gemm -lpapi -lamd_smi -lrocblas -lpthread
//
// Notes:
// - Everything else is left identical to your current file. The only changes are:
//     (1) removal of the gpu_power placeholder from PAPI CSV/stdout,
//     (2) per-kernel timing with HIP events and printing GFLOPS / TFLOPS,
//     (3) the compute backend switch: "rocblas" or "cus" (unchanged).
// - rocBLAS is fed your row-major A,B,C by doing the usual transpose trick:
//     (C^T) = (B^T) * (A^T) in column-major so results match your kernel.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>

#include <unistd.h>
#include <sys/time.h>

#include "hip/hip_runtime.h"
#include "papi.h"
#include <amd_smi/amdsmi.h>
#include <rocblas.h>

// ===================== Tunables =====================
#define M_DIM 14592
#define K_DIM 65536
#define N_DIM 14592

#define NUM_STREAMS 1
#define ITERATIONS_PER_STREAM 1

static int DEFAULT_DEVICE = 0;               // you can still override via argv/--device
static int DEFAULT_MONITOR_US = 300000;      // 300 ms polling

// ===================== Utility ======================
static double now_seconds()
{
    static timeval t0 = [](){ timeval x; gettimeofday(&x,nullptr); return x; }();
    timeval t; gettimeofday(&t,nullptr);
    return (t.tv_sec - t0.tv_sec) + (t.tv_usec - t0.tv_usec)/1e6;
}

static void usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s papi  rocblas [device]\n", prog);
    printf("  %s papi  cus     [device]\n", prog);
    printf("  %s amd   rocblas [device]\n", prog);
    printf("  %s amd   cus     [device]\n", prog);
    printf("  %s help\n", prog);
    printf("\nExamples:\n");
    printf("  %s papi  rocblas --device 0\n", prog);
    printf("  %s amd   cus 1\n", prog);
}

// ===================== Kernel =======================
__global__ void dgemm_kernel(const double *A, const double *B, double *C,
                             int M, int N, int K, double alpha, double beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// =============== Monitor Abstraction =================
struct Monitor {
    virtual ~Monitor() {}
    virtual bool start(FILE* csv, int device_index, int interval_us) = 0;
    virtual void stop() = 0;
};

// ---------------- PAPI Monitor -----------------------
struct PapiMonitor : public Monitor {
    std::atomic<bool> stop_{false};
    std::thread th_;
    int EventSet_ = PAPI_NULL;
    std::vector<const char*> names_;

    bool start(FILE* csv, int device_index, int interval_us) override {
        char buf[128];
        auto name = [&](const char* fmt)->const char*{
            static std::vector<std::string> store;
            snprintf(buf, sizeof(buf), fmt, device_index);
            store.emplace_back(buf);
            return store.back().c_str();
        };

        names_ = {
            name("amd_smi:::temp_current:device=%d:sensor=0"),
            name("amd_smi:::temp_current:device=%d:sensor=1"),
            name("amd_smi:::temp_current:device=%d:sensor=2"),
            name("amd_smi:::temp_current:device=%d:sensor=3"),
            name("amd_smi:::temp_current:device=%d:sensor=4"),
            name("amd_smi:::temp_current:device=%d:sensor=5"),
            name("amd_smi:::temp_current:device=%d:sensor=6"),
            name("amd_smi:::temp_current:device=%d:sensor=7"),
            name("amd_smi:::gfx_activity:device=%d"),
            name("amd_smi:::umc_activity:device=%d"),
            name("amd_smi:::mm_activity:device=%d"),
            name("amd_smi:::power_average:device=%d"),
            name("amd_smi:::power_cap_range_min:device=%d"),
            name("amd_smi:::power_cap_range_max:device=%d"),
            name("amd_smi:::mem_usage_VRAM:device=%d"),
            name("amd_smi:::mem_total_VRAM:device=%d"),
            name("amd_smi:::clk_freq_sys_current:device=%d"),
        };

        int st = PAPI_library_init(PAPI_VER_CURRENT);
        if (st != PAPI_VER_CURRENT) {
            fprintf(stderr, "PAPI init failed: %s\n", PAPI_strerror(st));
            return false;
        }
        st = PAPI_create_eventset(&EventSet_);
        if (st != PAPI_OK) { fprintf(stderr, "PAPI_create_eventset failed: %s\n", PAPI_strerror(st)); return false; }
        for (size_t i=0;i<names_.size();++i) {
            st = PAPI_add_named_event(EventSet_, names_[i]);
            if (st != PAPI_OK) { fprintf(stderr, "PAPI_add_named_event[%zu] (%s) failed: %s\n", i, names_[i], PAPI_strerror(st)); return false; }
        }
        st = PAPI_start(EventSet_);
        if (st != PAPI_OK) { fprintf(stderr, "PAPI_start failed: %s\n", PAPI_strerror(st)); return false; }

        // CSV header (gpu_power removed)
        fprintf(csv, "timestamp");
        for (auto n : names_) fprintf(csv, ",%s", n);
        fprintf(csv, "\n");
        fflush(csv);

        th_ = std::thread([=]() {
            std::vector<long long> values(names_.size(), 0);
            while (!stop_.load(std::memory_order_relaxed)) {
                int s = PAPI_read(EventSet_, values.data());
                if (s != PAPI_OK) {
                    fprintf(stderr, "PAPI_read failed: %s\n", PAPI_strerror(s));
                    break;
                }
                double t = now_seconds();

                // CSV line (no gpu_power)
                fprintf(csv, "%.6f", t);
                for (auto v : values) fprintf(csv, ",%lld", v);
                fprintf(csv, "\n");
                fflush(csv);

                // stdout (compact; no P=...)
                fprintf(stdout, "Time %.3f |", t);
                for (size_t i=0;i<values.size();++i) {
                    fprintf(stdout, " e%zu=%lld", i, values[i]);
                }
                fprintf(stdout, "\n");

                std::this_thread::sleep_for(std::chrono::microseconds(interval_us));
            }
        });
        return true;
    }

    void stop() override {
        stop_.store(true, std::memory_order_relaxed);
        if (th_.joinable()) th_.join();
        if (EventSet_ != PAPI_NULL) {
            PAPI_stop(EventSet_, nullptr);
            PAPI_cleanup_eventset(EventSet_);
            PAPI_destroy_eventset(&EventSet_);
            EventSet_ = PAPI_NULL;
        }
    }
};

// ---------------- AMD SMI Monitor --------------------
struct AmdSmiMonitor : public Monitor {
    std::atomic<bool> stop_{false};
    std::thread th_;
    amdsmi_processor_handle target_ = nullptr;

    struct Col { const char* name; };
    std::vector<Col> cols_;

    static amdsmi_processor_handle find_gpu_by_logical_index(uint32_t idx) {
        amdsmi_status_t st;
        uint32_t sock_count = 0;
        st = amdsmi_get_socket_handles(&sock_count, nullptr);
        if (st != AMDSMI_STATUS_SUCCESS || sock_count == 0) return nullptr;

        std::vector<amdsmi_socket_handle> sockets(sock_count);
        st = amdsmi_get_socket_handles(&sock_count, sockets.data());
        if (st != AMDSMI_STATUS_SUCCESS) return nullptr;

        uint32_t seen = 0;
        for (uint32_t s=0; s<sock_count; ++s) {
            uint32_t dev_count = 0;
            st = amdsmi_get_processor_handles(sockets[s], &dev_count, nullptr);
            if (st != AMDSMI_STATUS_SUCCESS || dev_count == 0) continue;
            std::vector<amdsmi_processor_handle> ph(dev_count);
            st = amdsmi_get_processor_handles(sockets[s], &dev_count, ph.data());
            if (st != AMDSMI_STATUS_SUCCESS) continue;

            for (uint32_t i=0; i<dev_count; ++i) {
                processor_type_t ty;
                if (amdsmi_get_processor_type(ph[i], &ty) == AMDSMI_STATUS_SUCCESS &&
                    ty == AMDSMI_PROCESSOR_TYPE_AMD_GPU)
                {
                    if (seen == idx) return ph[i];
                    ++seen;
                }
            }
        }
        return nullptr;
    }

    bool start(FILE* csv, int device_index, int interval_us) override {
        if (amdsmi_init(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS) {
            fprintf(stderr, "AMD SMI init failed\n");
            return false;
        }
        target_ = find_gpu_by_logical_index(device_index);
        if (!target_) {
            fprintf(stderr, "AMD SMI: could not find GPU handle for logical index %d\n", device_index);
            amdsmi_shut_down();
            return false;
        }

        cols_ = {
            {"Temp_Edge_C"}, {"Temp_Hotspot_C"}, {"Temp_VRAM_C"},
            {"GfxActivity_%"}, {"UmcActivity_%"}, {"MmActivity_%"},
            {"AvgSocketPower_W"}, {"CurrentSocketPower_W"},
            {"SelectedClk_SYS_MHz"}, {"SelectedClk_MEM_MHz"},
            {"ActualAvgClk_GFX_MHz"}, {"ActualAvgClk_MEM_MHz"},
            {"VramTotal_Bytes"}, {"VramUsed_Bytes"},
            {"FanSpeed_RPM"},
            {"PCIe_CurrentWidth"}, {"PCIe_CurrentSpeed_x0.1GTs"}, {"PCIe_CurrentBandwidth_Mbps"},
            {"ThrottleMask"}, {"IndepThrottleMask"},
            {"ProchotResidency_ns"}, {"PptResidency_ns"},
            {"SocketThmResidency_ns"}, {"VrThmResidency_ns"}, {"HbmThmResidency_ns"},
            {"AccumCounter_ns"}
        };

        fprintf(csv, "timestamp");
        for (auto &c : cols_) fprintf(csv, ",%s", c.name);
        fprintf(csv, "\n"); fflush(csv);

        th_ = std::thread([=]() {
            while (!stop_.load(std::memory_order_relaxed)) {
                double t = now_seconds();
                std::vector<long long> v(cols_.size(), -99);

                auto temp = [&](amdsmi_temperature_type_t ty)->long long {
                    int64_t value = -1;
                    amdsmi_get_temp_metric(target_, ty, AMDSMI_TEMP_CURRENT, &value);
                    return (long long)value;
                };
                v[0] = temp(AMDSMI_TEMPERATURE_TYPE_EDGE);
                v[1] = temp(AMDSMI_TEMPERATURE_TYPE_HOTSPOT);
                v[2] = temp(AMDSMI_TEMPERATURE_TYPE_VRAM);

                amdsmi_engine_usage_t act{};
                if (amdsmi_get_gpu_activity(target_, &act) == AMDSMI_STATUS_SUCCESS) {
                    v[3] = act.gfx_activity;
                    v[4] = act.umc_activity;
                    v[5] = act.mm_activity;
                }

                amdsmi_power_info_t pwr{};
                if (amdsmi_get_power_info(target_, &pwr) == AMDSMI_STATUS_SUCCESS) {
                    v[6] = (long long)pwr.average_socket_power;
                    v[7] = (long long)pwr.current_socket_power;
                }

                auto clk_selected = [&](amdsmi_clk_type_t ty)->long long {
                    amdsmi_frequencies_t f{};
                    if (amdsmi_get_clk_freq(target_, ty, &f) == AMDSMI_STATUS_SUCCESS) {
                        if (f.current < f.num_supported) {
                            uint64_t hz = f.frequency[f.current];
                            return (long long)(hz/1000000ULL);
                        }
                    }
                    return (long long)-1;
                };
                v[8]  = clk_selected(AMDSMI_CLK_TYPE_SYS);
                v[9]  = clk_selected(AMDSMI_CLK_TYPE_MEM);

                auto clk_avg = [&](amdsmi_clk_type_t ty)->long long {
                    amdsmi_clk_info_t ci{};
                    if (amdsmi_get_clock_info(target_, ty, &ci) == AMDSMI_STATUS_SUCCESS) {
                        return (long long)ci.clk; // MHz
                    }
                    return (long long)-1;
                };
                v[10] = clk_avg(AMDSMI_CLK_TYPE_GFX);
                v[11] = clk_avg(AMDSMI_CLK_TYPE_MEM);

                {
                    uint64_t total=0, used=0;
                    if (amdsmi_get_gpu_memory_total(target_, AMDSMI_MEM_TYPE_VRAM, &total) == AMDSMI_STATUS_SUCCESS)
                        v[12] = (long long)total;
                    if (amdsmi_get_gpu_memory_usage(target_, AMDSMI_MEM_TYPE_VRAM, &used) == AMDSMI_STATUS_SUCCESS)
                        v[13] = (long long)used;
                }

                {
                    int64_t rpm=-1;
                    if (amdsmi_get_gpu_fan_rpms(target_, 0, &rpm) == AMDSMI_STATUS_SUCCESS) v[14] = (long long)rpm;
                }

                {
                    amdsmi_pcie_info_t p{};
                    if (amdsmi_get_pcie_info(target_, &p) == AMDSMI_STATUS_SUCCESS) {
                        v[15] = p.pcie_metric.pcie_width;
                        v[16] = p.pcie_metric.pcie_speed;       // 0.1 GT/s
                        v[17] = p.pcie_metric.pcie_bandwidth;   // Mbps
                    }
                }

                {
                    amdsmi_gpu_metrics_t g{};
                    if (amdsmi_get_gpu_metrics_info(target_, &g) == AMDSMI_STATUS_SUCCESS) {
                        v[18] = (long long)g.throttle_status;
                        v[19] = (long long)g.indep_throttle_status;
                        v[20] = (long long)g.prochot_residency_acc;
                        v[21] = (long long)g.ppt_residency_acc;
                        v[22] = (long long)g.socket_thm_residency_acc;
                        v[23] = (long long)g.vr_thm_residency_acc;
                        v[24] = (long long)g.hbm_thm_residency_acc;
                        v[25] = (long long)g.accumulation_counter;
                    }
                }

                fprintf(csv, "%.6f", t);
                for (auto x : v) fprintf(csv, ",%lld", x);
                fprintf(csv, "\n"); fflush(csv);

                fprintf(stdout, "Time %.3f | Tedge=%lldC GfxAct=%lld%% Pavg=%lldW VRAM %lld/%lld MB\n",
                        t, v[0], v[3], v[6], (v[13]>>20), (v[12]>>20));

                std::this_thread::sleep_for(std::chrono::microseconds(interval_us));
            }
        });

        return true;
    }

    void stop() override {
        stop_.store(true, std::memory_order_relaxed);
        if (th_.joinable()) th_.join();
        amdsmi_shut_down();
    }
};

// =============== Modes ==============================
enum class MonitorMode { Papi, Amd, Help, Unknown };
enum class ComputeMode { Custom, Rocblas, Unknown };

static MonitorMode parse_monitor(const std::string& s) {
    if (s == "papi") return MonitorMode::Papi;
    if (s == "amd")  return MonitorMode::Amd;
    if (s == "help") return MonitorMode::Help;
    return MonitorMode::Unknown;
}
static ComputeMode parse_compute(const std::string& s) {
    if (s == "rocblas") return ComputeMode::Rocblas;
    if (s == "cus" || s == "custom" || s == "naive") return ComputeMode::Custom;
    return ComputeMode::Unknown;
}

// Parse device from any later arg: either "--device N" or a bare integer.
// Leaves your changed "hipSetDevice(1)" line untouched.
static bool is_integer(const std::string& s) {
    if (s.empty()) return false;
    size_t i = (s[0]=='+'||s[0]=='-') ? 1 : 0;
    if (i>=s.size()) return false;
    for (; i<s.size(); ++i) if (s[i]<'0'||s[i]>'9') return false;
    return true;
}
static bool parse_device_arg(int argc, char** argv, int& out_dev) {
    for (int i=2; i<argc; ++i) {
        std::string a = argv[i];
        if (a == "--device" && i+1<argc) { out_dev = std::atoi(argv[i+1]); return true; }
        if (a=="papi"||a=="amd"||a=="help"||a=="rocblas"||a=="cus"||a=="custom"||a=="naive") continue;
        if (is_integer(a)) { out_dev = std::atoi(a.c_str()); return true; }
    }
    return false;
}

static Monitor* make_monitor(MonitorMode m) {
    switch (m) {
        case MonitorMode::Papi: return new PapiMonitor();
        case MonitorMode::Amd:  return new AmdSmiMonitor();
        default: return nullptr;
    }
}

// =============== Compute helpers ====================
// Your original compute path (unchanged)
static void do_custom_dgemm(hipStream_t stream, const double* dA, const double* dB, double* dC,
                            int M, int N, int K, double alpha, double beta)
{
    dim3 block(32,32);
    dim3 grid((N + block.x - 1)/block.x,
              (M + block.y - 1)/block.y);
    hipLaunchKernelGGL(dgemm_kernel, grid, block, 0, stream,
                       dA, dB, dC, M, N, K, alpha, beta);
}

// rocBLAS DGEMM that matches your row-major semantics without touching your data.
// We compute (C^T) = (B^T) * (A^T) with column-major rocBLAS.
static bool do_rocblas_dgemm(rocblas_handle handle, hipStream_t stream,
                             const double* dA, const double* dB, double* dC,
                             int M, int N, int K, double alpha, double beta)
{
    rocblas_status rs = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    if (rs != rocblas_status_success) { fprintf(stderr, "rocBLAS set_pointer_mode failed (%d)\n", (int)rs); return false; }
    rs = rocblas_set_stream(handle, stream);
    if (rs != rocblas_status_success) { fprintf(stderr, "rocBLAS set_stream failed (%d)\n", (int)rs); return false; }

    const int m_prime = N;            // rows of C^T
    const int n_prime = M;            // cols of C^T
    const int k_prime = K;

    const int lda = m_prime;          // A' = B^T has shape (N x K)
    const int ldb = k_prime;          // B' = A^T has shape (K x M)
    const int ldc = m_prime;          // C' = C^T has shape (N x M)

    rs = rocblas_dgemm(handle,
                       rocblas_operation_none,   // A' not transposed (already B^T in memory)
                       rocblas_operation_none,   // B' not transposed (already A^T in memory)
                       m_prime, n_prime, k_prime,
                       &alpha,
                       dB, lda,    // A' points at B buffer
                       dA, ldb,    // B' points at A buffer
                       &beta,
                       dC, ldc);   // C' is C buffer (interpreted as column-major transpose)
    if (rs != rocblas_status_success) {
        fprintf(stderr, "rocBLAS dgemm failed (%d)\n", (int)rs);
        return false;
    }
    return true;
}

// ==================== Main ==========================
int main(int argc, char** argv)
{
    if (argc < 3) { usage(argv[0]); return 1; }
    MonitorMode mon_mode = parse_monitor(argv[1]);
    ComputeMode cmp_mode = parse_compute(argv[2]);
    if (mon_mode == MonitorMode::Help || mon_mode == MonitorMode::Unknown || cmp_mode == ComputeMode::Unknown) {
        usage(argv[0]); return (mon_mode==MonitorMode::Help?0:1);
    }

    int device_index = DEFAULT_DEVICE;
    parse_device_arg(argc, argv, device_index);

    // HIP device select — left IDENTICAL to your edited file.
    ///////////////////////////////////////
    ///////////////////////////////////////
        ///////////////////////////////////////
    ///////////////////////////////////////
        ///////////////////////////////////////
    ///////////////////////////////////////
        ///////////////////////////////////////
    ///////////////////////////////////////
        ///////////////////////////////////////
    ///////////////////////////////////////
        ///////////////////////////////////////
    ///////////////////////////////////////
        ///////////////////////////////////////
    ///////////////////////////////////////

    hipError_t hst = hipSetDevice(1);
    if (hst != hipSuccess) {
        fprintf(stderr, "hipSetDevice(%d) failed: %s\n", device_index, hipGetErrorString(hst));
        return 1;
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device_index) == hipSuccess) {
        printf("HIP Using Device %d: %s | CUs=%d | MaxThreadsPerBlock=%d\n",
               device_index, prop.name, prop.multiProcessorCount, prop.maxThreadsPerBlock);
    }

    // Host alloc
    size_t size_A = (size_t)M_DIM * K_DIM * sizeof(double);
    size_t size_B = (size_t)K_DIM * N_DIM * sizeof(double);
    size_t size_C = (size_t)M_DIM * N_DIM * sizeof(double);

    double *h_A=nullptr, *h_B=nullptr, *h_C=nullptr;
    hipHostMalloc(&h_A, size_A, hipHostMallocDefault);
    hipHostMalloc(&h_B, size_B, hipHostMallocDefault);
    hipHostMalloc(&h_C, size_C, hipHostMallocDefault);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host allocation failed\n"); return 1;
    }

    for (size_t i=0;i<(size_t)M_DIM*K_DIM;++i) h_A[i] = (double)(i%100);
    for (size_t i=0;i<(size_t)K_DIM*N_DIM;++i) h_B[i] = (double)(i%100);
    for (size_t i=0;i<(size_t)M_DIM*N_DIM;++i) h_C[i] = 0.0;

    // Device alloc + streams
    double *d_A[NUM_STREAMS], *d_B[NUM_STREAMS], *d_C[NUM_STREAMS];
    hipStream_t streams[NUM_STREAMS];
    hipEvent_t  events[NUM_STREAMS];

    // NEW: timing events per stream (to compute GFLOPS)
    hipEvent_t kstart[NUM_STREAMS], kstop[NUM_STREAMS];

    for (int s=0;s<NUM_STREAMS;++s) {
        if (hipMalloc(&d_A[s], size_A) != hipSuccess) { fprintf(stderr,"hipMalloc d_A[%d]\n",s); return 1; }
        if (hipMalloc(&d_B[s], size_B) != hipSuccess) { fprintf(stderr,"hipMalloc d_B[%d]\n",s); return 1; }
        if (hipMalloc(&d_C[s], size_C) != hipSuccess) { fprintf(stderr,"hipMalloc d_C[%d]\n",s); return 1; }
        if (hipStreamCreateWithFlags(&streams[s], hipStreamNonBlocking) != hipSuccess) { fprintf(stderr,"hipStreamCreate\n"); return 1; }
        if (hipEventCreate(&events[s]) != hipSuccess) { fprintf(stderr,"hipEventCreate\n"); return 1; }

        // create timing events
        if (hipEventCreate(&kstart[s]) != hipSuccess) { fprintf(stderr,"hipEventCreate kstart\n"); return 1; }
        if (hipEventCreate(&kstop[s])  != hipSuccess) { fprintf(stderr,"hipEventCreate kstop\n");  return 1; }
    }

    // H2D copies
    for (int s=0;s<NUM_STREAMS;++s) {
        if (hipMemcpyAsync(d_A[s], h_A, size_A, hipMemcpyHostToDevice, streams[s]) != hipSuccess) { fprintf(stderr,"H2D A\n"); return 1; }
        if (hipMemcpyAsync(d_B[s], h_B, size_B, hipMemcpyHostToDevice, streams[s]) != hipSuccess) { fprintf(stderr,"H2D B\n"); return 1; }
        if (hipMemcpyAsync(d_C[s], h_C, size_C, hipMemcpyHostToDevice, streams[s]) != hipSuccess) { fprintf(stderr,"H2D C\n"); return 1; }
    }
    for (int s=0;s<NUM_STREAMS;++s) hipStreamSynchronize(streams[s]);

    // CSV file
    const char* outname = (mon_mode==MonitorMode::Papi? "gemm_monitor_papi.csv":"gemm_monitor_amdsmi.csv");
    FILE* csv = fopen(outname, "w");
    if (!csv) { fprintf(stderr, "Failed to open %s\n", outname); return 1; }

    // Start monitor
    std::unique_ptr<Monitor> mon(make_monitor(mon_mode));
    if (!mon) { fclose(csv); return 1; }
    if (!mon->start(csv, device_index, DEFAULT_MONITOR_US)) {
        fprintf(stderr, "Failed to start monitor\n");
        fclose(csv);
        return 1;
    }

    // Compute selection
    const double alpha = 0.75, beta = 0.5;
    rocblas_handle handle = nullptr;
    if (cmp_mode == ComputeMode::Rocblas) {
        if (rocblas_create_handle(&handle) != rocblas_status_success) {
            fprintf(stderr, "rocBLAS create_handle failed\n");
            mon->stop(); fclose(csv); return 1;
        }
    }

    // FLOPs for DGEMM
    const double FLOP_COUNT = 2.0 * (double)M_DIM * (double)N_DIM * (double)K_DIM;
    double total_ms = 0.0;
    int total_calls = 0;

    // Launch compute (kept identical structure/timing, just bracketing with events)
    for (int iter=0; iter<ITERATIONS_PER_STREAM; ++iter) {
        for (int s=0; s<NUM_STREAMS; ++s) {
            hipEventRecord(kstart[s], streams[s]);  // start timing

            if (cmp_mode == ComputeMode::Custom) {
                do_custom_dgemm(streams[s], d_A[s], d_B[s], d_C[s], M_DIM, N_DIM, K_DIM, alpha, beta);
            } else {
                if (!do_rocblas_dgemm(handle, streams[s], d_A[s], d_B[s], d_C[s], M_DIM, N_DIM, K_DIM, alpha, beta)) {
                    fprintf(stderr, "rocBLAS compute failed\n");
                    if (handle) rocblas_destroy_handle(handle);
                    mon->stop(); fclose(csv); return 1;
                }
            }

            hipEventRecord(kstop[s], streams[s]);   // stop timing
            hipEventRecord(events[s], streams[s]);  // your original event
            hipStreamSynchronize(streams[s]);       // wait

            float ms = 0.0f;
            hipEventElapsedTime(&ms, kstart[s], kstop[s]);

            double secs = ms / 1000.0;
            double gflops = (FLOP_COUNT / 1e9) / secs;
            double tflops = gflops / 1000.0;

            printf("DGEMM[%s] time: %.3f ms  ->  %.3f GFLOPS  (%.3f TFLOPS)\n",
                   (cmp_mode == ComputeMode::Custom ? "cus" : "rocblas"),
                   ms, gflops, tflops);

            total_ms += ms;
            total_calls += 1;

            usleep(3000000); // match your pacing
        }
    }
    hipStreamSynchronize(streams[0]);

    if (total_calls > 0) {
        double avg_ms = total_ms / (double)total_calls;
        double avg_secs = avg_ms / 1000.0;
        double avg_gflops = (FLOP_COUNT / 1e9) / avg_secs;
        double avg_tflops = avg_gflops / 1000.0;
        printf("DGEMM avg over %d call(s): %.3f ms  ->  %.3f GFLOPS  (%.3f TFLOPS)\n",
               total_calls, avg_ms, avg_gflops, avg_tflops);
    }

    if (handle) rocblas_destroy_handle(handle);

    // Stop monitor & cleanup
    mon->stop();
    fclose(csv);

    for (int s=0;s<NUM_STREAMS;++s) {
        hipEventDestroy(kstart[s]);
        hipEventDestroy(kstop[s]);
        hipEventDestroy(events[s]);
        hipStreamDestroy(streams[s]);
        hipFree(d_A[s]); hipFree(d_B[s]); hipFree(d_C[s]);
    }
    hipHostFree(h_A); hipHostFree(h_B); hipHostFree(h_C);

    printf("Done. Wrote %s\n", outname);
    return 0;
}
