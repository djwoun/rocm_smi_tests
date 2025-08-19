// modular.cpp ? unified GEMM + modular monitors
// Modes:
//   Implementations: 'row' (GPU), 'col' (GPU), 'rocblas' (GPU)
//                    also available: 'row-cpu', 'col-cpu' (optional CPU refs)
//   Monitors:        'papi', 'amd', 'rocm', 'cli'
// Usage:
//   ./modular <implementation> <monitor> [M N K] [--print]
//
// Example:
//   ./modular row papi 4096 4096 4096 --print
//   ./modular col amd 8192 8192 8192 --print
//   ./modular rocblas papi 8192 8192 8192 --print
//
// Build (Makefile example matches your environment):
//   HIPCC = hipcc
//   PAPI ?= /storage/users/dwoun/apps/papi
//   INC = -I${PAPI}/include
//   LIB = -L${PAPI}/lib -lpapi -L/opt/rocm-6.4.0/lib -lpthread -lrocblas -lamd_smi -lrocm_smi64
//   CFLAGS = -O2 -Wall
//   %: %.cpp
//     $(HIPCC) $(CFLAGS) $(INC) $(LIB) -Wl,-rpath,$(PAPI)/lib -o $@ $<

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>   // strstr, sscanf, etc.
#include <cmath>

// HIP / rocBLAS
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// PAPI
#include <papi.h>

// AMD SMI (new) + ROCm SMI (legacy)
#include <amd_smi/amdsmi.h>
#include <rocm_smi/rocm_smi.h>

// ----------------- Helpers -----------------
#define HIP_CHECK(cmd) do { \
  hipError_t _e = (cmd); \
  if (_e != hipSuccess) { \
    std::cerr << "HIP error: " << hipGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

#define ROCBLAS_CHECK(cmd) do { \
  rocblas_status _s = (cmd); \
  if (_s != rocblas_status_success) { \
    std::cerr << "rocBLAS error: " << rocblas_status_to_string(_s) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

// ----------------- Interfaces -----------------
class IDgemm {
public:
  virtual ~IDgemm() = default;
  virtual void initialize(size_t M, size_t N, size_t K) = 0;
  virtual void run() = 0;
  virtual void report(double total_time_sec) const = 0;
  virtual void cleanup() = 0;
protected:
  size_t M=0, N=0, K=0;
  std::vector<double> h_A, h_B, h_C;
};

class IPerfMonitor {
public:
  virtual ~IPerfMonitor() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void report() const = 0;
  void setPrintEnabled(bool v){ printEnabled = v; }
  void setName(const std::string& n){ name = n; }
protected:
  bool printEnabled=false;
  std::string name;
  timeval start_time{};
};

// ----------------- CPU DGEMM (optional) -----------------
class ColumnMajorCpu : public IDgemm {
public:
  void initialize(size_t m, size_t n, size_t k) override {
    M=m; N=n; K=k;
    h_A.resize(M*K);
    h_B.resize(K*N);
    h_C.assign(M*N, 0.0);
    for (size_t i=0;i<M*K;++i) h_A[i] = (double)(i % 100);//double(rand())/RAND_MAX;
    for (size_t i=0;i<K*N;++i) h_B[i] = (double)(i % 100);//double(rand())/RAND_MAX;
  }
  void run() override {
    for (size_t j=0;j<N;++j){
      for (size_t i=0;i<M;++i){
        double sum=0.0;
        for (size_t l=0;l<K;++l){
          sum += h_A[i + l*M] * h_B[l + j*K];
        }
        h_C[i + j*M] = sum;
      }
    }
  }
  void report(double t) const override {
    double gflops = (2.0*M*N*K)/(t*1e9);
    std::cout<<"Implementation: CPU Col-Major DGEMM\n"
             <<"  Dimensions: M="<<M<<", N="<<N<<", K="<<K<<"\n"
             <<"  Execution Time: "<<std::fixed<<std::setprecision(6)<<t<<" s\n"
             <<"  Performance: "<<std::setprecision(2)<<gflops<<" GFLOPS\n";
  }
  void cleanup() override { h_A.clear(); h_B.clear(); h_C.clear(); }
};

class RowMajorCpu : public IDgemm {
public:
  void initialize(size_t m, size_t n, size_t k) override {
    M=m; N=n; K=k;
    h_A.resize(M*K);
    h_B.resize(K*N);
    h_C.assign(M*N, 0.0);
    for (size_t i=0;i<M*K;++i) h_A[i] = (double)(i % 100);//double(rand())/RAND_MAX;
    for (size_t i=0;i<K*N;++i) h_B[i] = (double)(i % 100);//double(rand())/RAND_MAX;
  }
  void run() override {
    for (size_t i=0;i<M;++i){
      for (size_t j=0;j<N;++j){
        double sum=0.0;
        for (size_t l=0;l<K;++l){
          sum += h_A[i*K + l] * h_B[l*N + j];
        }
        h_C[i*N + j] = sum;
      }
    }
  }
  void report(double t) const override {
    double gflops = (2.0*M*N*K)/(t*1e9);
    std::cout<<"Implementation: CPU Row-Major DGEMM\n"
             <<"  Dimensions: M="<<M<<", N="<<N<<", K="<<K<<"\n"
             <<"  Execution Time: "<<std::fixed<<std::setprecision(6)<<t<<" s\n"
             <<"  Performance: "<<std::setprecision(2)<<gflops<<" GFLOPS\n";
  }
  void cleanup() override { h_A.clear(); h_B.clear(); h_C.clear(); }
};

// ----------------- GPU kernels (same logic as gemm.cpp) -----------------
__global__ void dgemm_kernel_rowmajor(const double *A, const double *B, double *C,
                                      int M, int N, int K, double alpha, double beta) {
    // Compute the row and column index of the C element.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        // Compute the dot product of row of A and column of B.
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
            //sum += sin(A[row * K + k] * B[k * N + col]) + cos(A[row * K + k] * B[k * N + col]);
        }
        // Scale the result and add the scaled C element.
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

__global__ void dgemm_kernel_colmajor(const double *A, const double *B, double *C,
                                      int M, int N, int K, double alpha, double beta) {
    // Compute the row and column index of the C element (i=row, j=col).
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (i < M && j < N) {
        double sum = 0.0;
        // Compute the dot product of column-major A(:,l) and B(l,:).
        for (int l = 0; l < K; l++) {
            sum += A[i + l * M] * B[l + j * K];
            //sum += sin(A[i + l * M] * B[l + j * K]) + cos(A[i + l * M] * B[l + j * K]);
        }
        // Scale the result and add the scaled C element.
        C[i + j * M] = alpha * sum + beta * C[i + j * M];
    }
}

// ----------------- GPU DGEMM (row-major) -----------------
class GpuRowDgemm : public IDgemm {
  double *dA=nullptr, *dB=nullptr, *dC=nullptr;
public:
  void initialize(size_t m, size_t n, size_t k) override {
    M=m; N=n; K=k;
    h_A.resize(M*K);
    h_B.resize(K*N);
    h_C.assign(M*N, 0.0);
    for (size_t i=0;i<M*K;++i) h_A[i] = (double)(i % 100); // double(rand())/RAND_MAX;
    for (size_t i=0;i<K*N;++i) h_B[i] = (double)(i % 100); //double(rand())/RAND_MAX;  // fixed '<'
  }
  void run() override {
    // allocate & copy on first run (lazy allocate)
    if (!dA) {
      HIP_CHECK(hipMalloc(&dA, M*K*sizeof(double)));
      HIP_CHECK(hipMalloc(&dB, K*N*sizeof(double)));
      HIP_CHECK(hipMalloc(&dC, M*N*sizeof(double)));
      HIP_CHECK(hipMemcpy(dA, h_A.data(), M*K*sizeof(double), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dB, h_B.data(), K*N*sizeof(double), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dC, 0, M*N*sizeof(double)));
    }
    const double alpha=0.75, beta=0.5;  // match gemm.cpp
    dim3 block(32,32);
    dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
    hipLaunchKernelGGL(dgemm_kernel_rowmajor, grid, block, 0, 0,
                       dA, dB, dC, (int)M, (int)N, (int)K, alpha, beta);
    HIP_CHECK(hipDeviceSynchronize());
  }
  void report(double t) const override {
    double gflops = (2.0*M*N*K)/(t*1e9);
    std::cout<<"Implementation: GPU Row-Major DGEMM (gemm.cpp logic)\n"
             <<"  Dimensions: M="<<M<<", N="<<N<<", K="<<K<<"\n"
             <<"  Execution Time: "<<std::fixed<<std::setprecision(6)<<t<<" s\n"
             <<"  Performance: "<<std::setprecision(2)<<gflops<<" GFLOPS\n";
  }
  void cleanup() override {
    if (dA) hipFree(dA);
    if (dB) hipFree(dB);
    if (dC) hipFree(dC);
    dA=dB=dC=nullptr; h_A.clear(); h_B.clear(); h_C.clear();
  }
};

// ----------------- GPU DGEMM (col-major) -----------------
class GpuColDgemm : public IDgemm {
  double *dA=nullptr, *dB=nullptr, *dC=nullptr;
public:
  void initialize(size_t m, size_t n, size_t k) override {
    M=m; N=n; K=k;
    h_A.resize(M*K);
    h_B.resize(K*N);
    h_C.assign(M*N, 0.0);
    for (size_t i=0;i<M*K;++i) h_A[i] = (double)(i % 100); // double(rand())/RAND_MAX;
    for (size_t i=0;i<K*N;++i) h_B[i] = (double)(i % 100); //double(rand())/RAND_MAX;
  }
  void run() override {
    if (!dA) {
      HIP_CHECK(hipMalloc(&dA, M*K*sizeof(double)));
      HIP_CHECK(hipMalloc(&dB, K*N*sizeof(double)));
      HIP_CHECK(hipMalloc(&dC, M*N*sizeof(double)));
      HIP_CHECK(hipMemcpy(dA, h_A.data(), M*K*sizeof(double), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dB, h_B.data(), K*N*sizeof(double), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemset(dC, 0, M*N*sizeof(double)));
    }
    const double alpha=0.75, beta=0.5;  // match gemm.cpp
    dim3 block(32,32);
    dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
    hipLaunchKernelGGL(dgemm_kernel_colmajor, grid, block, 0, 0,
                       dA, dB, dC, (int)M, (int)N, (int)K, alpha, beta);
    HIP_CHECK(hipDeviceSynchronize());
  }
  void report(double t) const override {
    double gflops = (2.0*M*N*K)/(t*1e9);
    std::cout<<"Implementation: GPU Col-Major DGEMM (gemm.cpp logic)\n"
             <<"  Dimensions: M="<<M<<", N="<<N<<", K="<<K<<"\n"
             <<"  Execution Time: "<<std::fixed<<std::setprecision(6)<<t<<" s\n"
             <<"  Performance: "<<std::setprecision(2)<<gflops<<" GFLOPS\n";
  }
  void cleanup() override {
    if (dA) hipFree(dA);
    if (dB) hipFree(dB);
    if (dC) hipFree(dC);
    dA=dB=dC=nullptr; h_A.clear(); h_B.clear(); h_C.clear();
  }
};

// ----------------- GPU DGEMM (rocBLAS) -----------------
class RocblasDgemm : public IDgemm {
  double *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
  rocblas_handle handle=nullptr;
public:
  void initialize(size_t m, size_t n, size_t k) override {
    M=m; N=n; K=k;
    h_A.resize(M*K); h_B.resize(K*N); h_C.assign(M*N, 0.0);
    for (size_t i=0;i<M*K;++i) h_A[i] = (double)(i % 100); //double(rand())/RAND_MAX;
    for (size_t i=0;i<K*N;++i) h_B[i] = (double)(i % 100); //double(rand())/RAND_MAX;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    HIP_CHECK(hipMalloc(&d_A, M*K*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B, K*N*sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C, M*N*sizeof(double)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M*K*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), K*N*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, M*N*sizeof(double)));
  }
  void run() override {
    const double alpha=1.0, beta=0.0;
    // rocBLAS is column-major; A(MxK), B(KxN), C(MxN), lda=M, ldb=K, ldc=M
    ROCBLAS_CHECK(rocblas_dgemm(handle,
                                rocblas_operation_none, rocblas_operation_none,
                                (int)M, (int)N, (int)K,
                                &alpha,
                                d_A, (int)M,
                                d_B, (int)K,
                                &beta,
                                d_C, (int)M));
    HIP_CHECK(hipDeviceSynchronize());
  }
  void report(double t) const override {
    double gflops = (2.0*M*N*K)/(t*1e9);
    std::cout<<"Implementation: rocBLAS GPU DGEMM\n"
             <<"  Dimensions: M="<<M<<", N="<<N<<", K="<<K<<"\n"
             <<"  Execution Time: "<<std::fixed<<std::setprecision(6)<<t<<" s\n"
             <<"  Performance: "<<std::setprecision(2)<<gflops<<" GFLOPS\n";
  }
  void cleanup() override {
    if (d_A) hipFree(d_A);
    if (d_B) hipFree(d_B);
    if (d_C) hipFree(d_C);
    if (handle) rocblas_destroy_handle(handle);
    d_A=d_B=d_C=nullptr; handle=nullptr;
    h_A.clear(); h_B.clear(); h_C.clear();
  }
};

// ----------------- Monitors -----------------
class PapiMonitor : public IPerfMonitor {
  int EventSet = PAPI_NULL;
  std::vector<const char*> ev;
  std::atomic<bool> stopFlag{false};
  std::thread thr;
  FILE* csv=nullptr;
public:
  PapiMonitor(){
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
      throw std::runtime_error("PAPI_library_init failed");
    if (PAPI_create_eventset(&EventSet) != PAPI_OK)
      throw std::runtime_error("PAPI_create_eventset failed");

    // Named events via ROCm SMI component (device 0)
    ev = {
      "rocm_smi:::temp_current:device=0:sensor=1",   // edge
      "rocm_smi:::temp_current:device=0:sensor=2",   // hotspot
      "rocm_smi:::mem_usage_VRAM:device=0",          // bytes
      "rocm_smi:::busy_percent:device=0",            // %
      "rocm_smi:::memory_busy_percent:device=0"      // %
    };
    for (auto* e : ev) {
      int rc = PAPI_add_named_event(EventSet, e);
      if (rc != PAPI_OK)
        throw std::runtime_error(std::string("PAPI_add_named_event failed: ") + e);
    }
  }
  void start() override {
    if (PAPI_start(EventSet) != PAPI_OK)
      throw std::runtime_error("PAPI_start failed");
    std::string fname = "gemm_" + name + ".csv";
    csv = fopen(fname.c_str(), "w");
    if (!csv) throw std::runtime_error("cannot open CSV for PAPI");
    fprintf(csv, "timestamp");
    for (auto* e: ev) fprintf(csv, ",%s", e);
    fprintf(csv, ",CLI_power_W\n"); fflush(csv);
    gettimeofday(&start_time, nullptr);
    stopFlag=false;
    thr = std::thread([this]{ loop(); });
  }
  void loop(){
    while (!stopFlag.load()){
      long long v[5]={0,0,0,0,0};
      if (PAPI_read(EventSet, v) != PAPI_OK){
        fprintf(stderr, "PAPI_read failed\n");
        break;
      }
      timeval now{}; gettimeofday(&now, nullptr);
      double t = (now.tv_sec - start_time.tv_sec) + (now.tv_usec - start_time.tv_usec)/1e6;

      // optional power via CLI (best-effort)
      int powerW = -1;
      FILE* fp = popen("amd-smi metric -g 0 -p --csv", "r");
      if (fp){
        char buf[256]; bool header=false;
        while (fgets(buf,sizeof(buf),fp)){
          if (!header && strstr(buf,"gpu")) { header=true; continue; }
          if (header){
            int gid=0, p=0;
            // heuristic: GPU,Power(W),...
            if (sscanf(buf, "%d,%d", &gid, &p) >= 2) { powerW=p; }
            break;
          }
        }
        pclose(fp);
      }

      fprintf(csv, "%.6f,%lld,%lld,%lld,%lld,%lld,%d\n", t, v[0],v[1],v[2],v[3],v[4], powerW);
      fflush(csv);
      if (printEnabled){
        std::cout<<std::fixed<<std::setprecision(3)
                 <<"t="<<t<<"s | Edge "<<v[0]<<" mC, Hot "<<v[1]
                 <<" mC, VRAM "<<v[2]<<", GPU "<<v[3]<<"%, MEM "<<v[4]
                 <<"%, Pwr "<<powerW<<" W\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
  void stop() override {
    stopFlag=true;
    if (thr.joinable()) thr.join();
    PAPI_stop(EventSet, nullptr);
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    if (csv) fclose(csv);
  }
  void report() const override {
    std::cout<<"Performance data saved to gemm_"<<name<<".csv\n";
  }
};

class AmdSmiMonitor : public IPerfMonitor {
  amdsmi_processor_handle h = nullptr;
  std::atomic<bool> stopFlag{false};
  std::thread thr;
  FILE* csv=nullptr;
public:
  AmdSmiMonitor(){
    if (amdsmi_init(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS)
      throw std::runtime_error("amdsmi_init failed");

    // first socket / first GPU
    uint32_t sock_cnt=0;
    if (amdsmi_get_socket_handles(&sock_cnt,nullptr) != AMDSMI_STATUS_SUCCESS || sock_cnt==0)
      throw std::runtime_error("no AMD GPU sockets");
    std::vector<amdsmi_socket_handle> socks(sock_cnt);
    amdsmi_get_socket_handles(&sock_cnt,socks.data());

    uint32_t proc_cnt=0;
    if (amdsmi_get_processor_handles(socks[0], &proc_cnt, nullptr) != AMDSMI_STATUS_SUCCESS || proc_cnt==0)
      throw std::runtime_error("no AMD GPU processors");
    std::vector<amdsmi_processor_handle> procs(proc_cnt);
    amdsmi_get_processor_handles(socks[0], &proc_cnt, procs.data());

    for (auto ph: procs){
      processor_type_t pt;
      if (amdsmi_get_processor_type(ph, &pt)==AMDSMI_STATUS_SUCCESS &&
          pt==AMDSMI_PROCESSOR_TYPE_AMD_GPU){ h=ph; break; }
    }
    if (!h) throw std::runtime_error("failed to get GPU handle");
  }
  void start() override {
    std::string fname = "gemm_" + name + ".csv";
    csv = fopen(fname.c_str(), "w");
    if (!csv) throw std::runtime_error("cannot open CSV for AMD SMI");
    fprintf(csv, "timestamp,Temp_Edge_mC,Temp_Hotspot_mC,GfxActivity_%%,UmcActivity_%%,VramTotal_B,VramUsed_B,AvgSocketPower_W,CurrentSocketPower_W\n");
    fflush(csv);
    gettimeofday(&start_time, nullptr);
    stopFlag=false;
    thr = std::thread([this]{ loop(); });
  }
  void loop(){
    while (!stopFlag.load()){
      timeval now{}; gettimeofday(&now,nullptr);
      double t = (now.tv_sec - start_time.tv_sec) + (now.tv_usec - start_time.tv_usec)/1e6;

      int64_t edge=-1, hot=-1;
      amdsmi_get_temp_metric(h, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &edge);
      amdsmi_get_temp_metric(h, AMDSMI_TEMPERATURE_TYPE_HOTSPOT, AMDSMI_TEMP_CURRENT, &hot);

      amdsmi_engine_usage_t act{};
      long long gfx=-1, umc=-1;
      if (amdsmi_get_gpu_activity(h,&act)==AMDSMI_STATUS_SUCCESS){
        gfx = (long long)act.gfx_activity;
        umc = (long long)act.umc_activity;
      }

      uint64_t vtot=0, vused=0;
      long long Ltot=-1, Lused=-1;
      if (amdsmi_get_gpu_memory_total(h, AMDSMI_MEM_TYPE_VRAM, &vtot)==AMDSMI_STATUS_SUCCESS) Ltot=(long long)vtot;
      if (amdsmi_get_gpu_memory_usage(h, AMDSMI_MEM_TYPE_VRAM, &vused)==AMDSMI_STATUS_SUCCESS) Lused=(long long)vused;

      amdsmi_power_info_t pinfo{};
      long long pavg=-1, pcurr=-1;
      if (amdsmi_get_power_info(h,&pinfo)==AMDSMI_STATUS_SUCCESS){
        pavg = (long long)pinfo.average_socket_power;
        pcurr= (long long)pinfo.current_socket_power;
      }

      fprintf(csv,"%.6f,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n",
              t,(long long)edge,(long long)hot,gfx,umc,Ltot,Lused,pavg,pcurr);
      fflush(csv);
      if (printEnabled){
        std::cout<<std::fixed<<std::setprecision(3)
                 <<"t="<<t<<"s | Edge "<<edge<<" mC, Hot "<<hot
                 <<" mC, GFX "<<gfx<<" %, UMC "<<umc<<" %, Pwr "<<pcurr<<" W\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
  void stop() override {
    stopFlag=true;
    if (thr.joinable()) thr.join();
    amdsmi_shut_down();
    if (csv) fclose(csv);
  }
  void report() const override {
    std::cout<<"Performance data saved to gemm_"<<name<<".csv\n";
  }
};

class RocmSmiMonitor : public IPerfMonitor {
  uint32_t dev=0;
  std::atomic<bool> stopFlag{false};
  std::thread thr;
  FILE* csv=nullptr;
public:
  RocmSmiMonitor(){
    if (rsmi_init(0) != RSMI_STATUS_SUCCESS)
      throw std::runtime_error("rsmi_init failed");
  }
  ~RocmSmiMonitor(){ rsmi_shut_down(); }
  void start() override {
    std::string fname = "gemm_" + name + ".csv";
    csv = fopen(fname.c_str(),"w");
    if (!csv) throw std::runtime_error("cannot open CSV for ROCm SMI");
    fprintf(csv, "timestamp,GPUUtil_%%,Power_W,Temp_Edge_mC\n"); fflush(csv);
    gettimeofday(&start_time,nullptr);
    stopFlag=false;
    thr = std::thread([this]{ loop(); });
  }
  void loop(){
    while (!stopFlag.load()){
      timeval now{}; gettimeofday(&now,nullptr);
      double t = (now.tv_sec - start_time.tv_sec) + (now.tv_usec - start_time.tv_usec)/1e6;

      uint32_t busy=0; if (rsmi_dev_busy_percent_get(dev,&busy)!=RSMI_STATUS_SUCCESS) busy=UINT32_MAX;
      uint64_t p_uW=0; long long pW=-1;
      if (rsmi_dev_power_ave_get(dev,0,&p_uW)==RSMI_STATUS_SUCCESS) pW = (long long)(p_uW/1000000.0);
      int64_t edge=0; long long edge_mc=-1;
      if (rsmi_dev_temp_metric_get(dev,RSMI_TEMP_TYPE_EDGE,RSMI_TEMP_CURRENT,&edge)==RSMI_STATUS_SUCCESS) edge_mc=edge;

      fprintf(csv,"%.6f,%u,%lld,%lld\n", t, busy, pW, edge_mc);
      fflush(csv);
      if (printEnabled){
        std::cout<<std::fixed<<std::setprecision(3)
                 <<"t="<<t<<"s | Util "<<(busy==UINT32_MAX?-1:(int)busy)<<" %, "
                 <<"Pwr "<<pW<<" W, Edge "<<edge_mc<<" mC\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
  void stop() override {
    stopFlag=true;
    if (thr.joinable()) thr.join();
    if (csv) fclose(csv);
  }
  void report() const override {
    std::cout<<"Performance data saved to gemm_"<<name<<".csv\n";
  }
};

class CliMonitor : public IPerfMonitor {
  std::atomic<bool> stopFlag{false};
  std::thread thr;
  FILE* csv=nullptr;
public:
  void start() override {
    std::string fname = "gemm_" + name + ".csv";
    csv = fopen(fname.c_str(),"w");
    if (!csv) throw std::runtime_error("cannot open CSV for CLI");
    fprintf(csv, "timestamp,GPU_Temp_mC,GPU_Power_W\n"); fflush(csv);
    gettimeofday(&start_time,nullptr);
    stopFlag=false;
    thr = std::thread([this]{ loop(); });
  }
  void loop(){
    while (!stopFlag.load()){
      timeval now{}; gettimeofday(&now,nullptr);
      double t = (now.tv_sec - start_time.tv_sec) + (now.tv_usec - start_time.tv_usec)/1e6;

      int temp_mC = -1, pwr_W = -1;
      FILE* fp = popen("amd-smi --showtemp --showpower --csv", "r");
      if (fp){
        char buf[256]; bool header=false;
        while (fgets(buf,sizeof(buf),fp)){
          if (!header && strstr(buf,"GPU")!=nullptr){ header=true; continue; }
          if (header){
            int gid=0; double tc=0.0, pw=0.0;
            if (sscanf(buf, "%d, %lf, %lf", &gid, &tc, &pw)==3){
              temp_mC = (int)std::lround(tc*1000.0);
              pwr_W   = (int)std::lround(pw);
            }
            break;
          }
        }
        pclose(fp);
      }

      fprintf(csv,"%.6f,%d,%d\n", t, temp_mC, pwr_W);
      fflush(csv);
      if (printEnabled){
        std::cout<<std::fixed<<std::setprecision(3)
                 <<"t="<<t<<"s | Temp "<<temp_mC<<" mC, Pwr "<<pwr_W<<" W\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
  void stop() override {
    stopFlag=true;
    if (thr.joinable()) thr.join();
    if (csv) fclose(csv);
  }
  void report() const override {
    std::cout<<"Performance data saved to gemm_"<<name<<".csv\n";
  }
};

// ----------------- Factories -----------------
class DgemmFactory {
public:
  static std::unique_ptr<IDgemm> create(const std::string& t){
    if (t=="row")      return std::make_unique<GpuRowDgemm>();       // GPU (row-major kernel)
    if (t=="col")      return std::make_unique<GpuColDgemm>();       // GPU (col-major kernel)
    if (t=="rocblas")  return std::make_unique<RocblasDgemm>();      // GPU (lib)
    if (t=="row-cpu")  return std::make_unique<RowMajorCpu>();       // optional CPU
    if (t=="col-cpu")  return std::make_unique<ColumnMajorCpu>();    // optional CPU
    throw std::invalid_argument("unknown implementation: "+t);
  }
};

class PerfMonitorFactory {
public:
  static std::unique_ptr<IPerfMonitor> create(const std::string& t){
    if (t=="papi") return std::make_unique<PapiMonitor>();
    if (t=="amd")  return std::make_unique<AmdSmiMonitor>();
    if (t=="rocm") return std::make_unique<RocmSmiMonitor>();
    if (t=="cli")  return std::make_unique<CliMonitor>();
    throw std::invalid_argument("unknown monitor: "+t);
  }
};

// ----------------- Main -----------------
int main(int argc, char** argv){
  auto usage = [&](const char* p){
    std::cerr<<"Usage: "<<p<<" <implementation> <monitor> [M N K] [--print]\n"
             <<"  <implementation>: 'row', 'col', 'rocblas'   (GPU)\n"
             <<"                    also: 'row-cpu', 'col-cpu' (CPU reference)\n"
             <<"  <monitor>:        'papi', 'amd', 'rocm', 'cli'\n"
             <<"  [M N K]:          optional matrix dims (default: 14592 14592 65536)\n"
             <<"  [--print]:        live-print samples while logging CSV\n";
  };
  if (argc < 3){ usage(argv[0]); return 1; }

  bool live=false;
  if (std::string(argv[argc-1])=="--print"){ live=true; --argc; }

  std::string impl = argv[1];
  std::string mon  = argv[2];

  size_t M=14592, N=14592, K=65536;  // keep large defaults to match your runs
  if (argc > 3){
    if (argc != 6){
      std::cerr<<"Error: provide all 3 dims (M N K) or none.\n";
      usage(argv[0]); return 1;
    }
    try {
      M = std::stoull(argv[3]);
      N = std::stoull(argv[4]);
      K = std::stoull(argv[5]);
    } catch (...) {
      std::cerr<<"Invalid dimensions.\n"; return 1;
    }
  }

  std::cout<<"Configuration:\n"
           <<"  Implementation: "<<impl<<"\n"
           <<"  Monitor: "<<mon<<"\n"
           <<"  Dimensions: M="<<M<<", N="<<N<<", K="<<K<<"\n"
           <<"----------------------------------------\n\n";

  try{
    auto gemm = DgemmFactory::create(impl);
    auto monu = PerfMonitorFactory::create(mon);
    monu->setName(mon);
    monu->setPrintEnabled(live);

    std::cout<<"Initializing...\n";
    gemm->initialize(M,N,K);

    std::cout<<"Starting computation...\n";
    monu->start();
    auto t0 = std::chrono::high_resolution_clock::now();
    gemm->run();
    auto t1 = std::chrono::high_resolution_clock::now();
    monu->stop();

    std::cout<<"Computation finished.\n\n--- Results Summary ---\n";
    std::chrono::duration<double> dt = t1 - t0;
    gemm->report(dt.count());
    monu->report();

    std::cout<<"\nCleaning up...\n";
    gemm->cleanup();
    std::cout<<"Done.\n";
  } catch (const std::exception& e){
    std::cerr<<"Fatal error: "<<e.what()<<"\n";
    return 1;
  }
  return 0;
}
