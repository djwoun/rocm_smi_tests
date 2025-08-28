// overhead_modular.cpp
// One binary, three selectable profiles: --set {key|all|pcie}.
// Keeps your PAPI vs AMDSMI timing style, --verify first-value prints,
// ROCm 6.4-safe SMI calls, and excludes clk_freq_level_2 + PCIe throughput
// from the "all" set while "pcie" isolates those.
//
// Build:
//   hipcc -O2 -Wall -I$PAPI/include -L$PAPI/lib -lpapi \
//         -L/opt/rocm-6.4.0/lib -lpthread -lamd_smi -o overhead_modular overhead_modular.cpp
//
// Examples:
//   ./overhead_modular --set key  --device 0 --iters 100 --out key.csv --verify
//   ./overhead_modular --set all  --device 0 --iters 100 --events a.txt --out all.csv
//   ./overhead_modular --set pcie --device 0 --iters 100 --out pcie.csv

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <algorithm>

#include <papi.h>
#include <amd_smi/amdsmi.h>

// ---------- tiny utils ----------
static inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
static inline bool starts_with(const std::string& s, const char* pfx) {
    size_t n = std::strlen(pfx);
    return s.size() >= n && std::memcmp(s.data(), pfx, n) == 0;
}
static inline bool is_space(char c){ return c==' '||c=='\t'||c=='\r'||c=='\n'; }

// ---------- args ----------
struct Args {
    enum Set { KEY, ALL, PCIE } which = KEY;
    size_t iters = 100;
    int device = 0;
    std::string out = "overhead.csv";
    std::string events_path = "a.txt"; // used by --set all
    bool verify = false;
    bool papi_only = false;
    bool smi_only = false;
};
static bool parse_args(int argc, char** argv, Args& a) {
    for (int i=1;i<argc;i++) {
        std::string s(argv[i]);
        if      (s=="--set"    && i+1<argc) {
            std::string v = argv[++i];
            if      (v=="key")  a.which = Args::KEY;
            else if (v=="all")  a.which = Args::ALL;
            else if (v=="pcie") a.which = Args::PCIE;
            else { fprintf(stderr,"Unknown set '%s' (key|all|pcie)\n", v.c_str()); return false; }
        }
        else if (s=="--iters"   && i+1<argc) a.iters = (size_t)strtoull(argv[++i],nullptr,10);
        else if (s=="--device"  && i+1<argc) a.device = atoi(argv[++i]);
        else if (s=="--out"     && i+1<argc) a.out = argv[++i];
        else if (s=="--events"  && i+1<argc) a.events_path = argv[++i];
        else if (s=="--verify"  ) a.verify = true;
        else if (s=="--papi-only") a.papi_only = true;
        else if (s=="--smi-only" ) a.smi_only = true;
        else { fprintf(stderr,"Unknown arg: %s\n", s.c_str()); return false; }
    }
    return true;
}

// ---------- CSV ----------
static void write_csv_header(FILE* f) {
    fprintf(f, "metric,source,api,iters,ok_count,min_us,avg_us,max_us,notes\n");
}
struct Stat {
    size_t n = 0, ok_count = 0;
    uint64_t min_ns = UINT64_MAX, max_ns = 0;
    __int128 sum_ns = 0;
};
static void write_csv_row(FILE* f, const char* source, const char* metric, const char* api,
                          size_t iters, const Stat& s, const std::string& notes) {
    double min_us = (s.min_ns==UINT64_MAX?0.0: (double)s.min_ns/1000.0);
    double max_us = (double)s.max_ns/1000.0;
    double avg_us = (s.ok_count? (double)((long double)s.sum_ns/(long double)s.ok_count)/1000.0 : 0.0);
    fprintf(f, "%s,%s,%s,%zu,%zu,%.3f,%.3f,%.3f,%s\n",
            metric, source, api, iters, s.ok_count, min_us, avg_us, max_us,
            notes.c_str());
}

// ---------- AMD SMI discovery ----------
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
                ty == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
                if (seen == idx) return ph[i];
                ++seen;
            }
        }
    }
    return nullptr;
}

// ---------- classification ----------
enum class Kind {
    TEMP_METRIC, GFX_ACTIVITY, UMC_ACTIVITY, MM_ACTIVITY,
    VRAM_TOTAL, VRAM_USED,
    POWER_AVG_W, POWER_CAP_CUR, POWER_CAP_MIN, POWER_CAP_MAX,
    CLK_SYS_CURRENT_MHZ, CLK_SYS_COUNT, CLK_SYS_LEVEL,
    PCIE_THROUGHPUT_SENT, PCIE_THROUGHPUT_RECEIVED, PCIE_THROUGHPUT_MAXPKT,
    PCIE_REPLAY_COUNTER, ENERGY_CONSUMED, GPU_BDFID, NUMA_NODE,
    GPU_ID, GPU_REVISION, GPU_SUBSYSTEM_ID
};
struct MetricDef { std::string name; Kind kind; int arg=0; int arg2=0; };

static const amdsmi_temperature_type_t kTempSensorMap[8] = {
    AMDSMI_TEMPERATURE_TYPE_EDGE,
    AMDSMI_TEMPERATURE_TYPE_JUNCTION, // hotspot
    AMDSMI_TEMPERATURE_TYPE_VRAM,
    AMDSMI_TEMPERATURE_TYPE_HBM_0,
    AMDSMI_TEMPERATURE_TYPE_HBM_1,
    AMDSMI_TEMPERATURE_TYPE_HBM_2,
    AMDSMI_TEMPERATURE_TYPE_HBM_3,
    AMDSMI_TEMPERATURE_TYPE_PLX
};
static bool temp_metric_from_string(const std::string& nm, amdsmi_temperature_metric_t* out) {
    struct Pair { const char* k; amdsmi_temperature_metric_t v; };
    static const Pair map[] = {
        {"temp_current", AMDSMI_TEMP_CURRENT}, {"temp_max", AMDSMI_TEMP_MAX},
        {"temp_min", AMDSMI_TEMP_MIN}, {"temp_max_hyst", AMDSMI_TEMP_MAX_HYST},
        {"temp_min_hyst", AMDSMI_TEMP_MIN_HYST}, {"temp_critical", AMDSMI_TEMP_CRITICAL},
        {"temp_critical_hyst", AMDSMI_TEMP_CRITICAL_HYST}, {"temp_emergency", AMDSMI_TEMP_EMERGENCY},
        {"temp_emergency_hyst", AMDSMI_TEMP_EMERGENCY_HYST}, {"temp_crit_min", AMDSMI_TEMP_CRIT_MIN},
        {"temp_crit_min_hyst", AMDSMI_TEMP_CRIT_MIN_HYST}, {"temp_offset", AMDSMI_TEMP_OFFSET},
        {"temp_lowest", AMDSMI_TEMP_LOWEST}, {"temp_highest", AMDSMI_TEMP_HIGHEST},
    };
    for (auto &kv : map) if (nm == kv.k) { *out = kv.v; return true; }
    return false;
}

// drop logic for "all": exclude PCIe throughput & unsupported clk_freq_level_2
static bool should_drop_event_all(const std::string& ev, int dev) {
    if (starts_with(ev, "amd_smi:::pci_throughput_sent:")
     || starts_with(ev, "amd_smi:::pci_throughput_received:")
     || starts_with(ev, "amd_smi:::pci_throughput_max_packet:")
     || starts_with(ev, "rocm_smi:::pci_throughput_sent:")
     || starts_with(ev, "rocm_smi:::pci_throughput_received:")
     || starts_with(ev, "rocm_smi:::pci_throughput_max_packet:"))
        return true;
    const std::string ban = "amd_smi:::clk_freq_level_2:device=" + std::to_string(dev);
    if (ev == ban) return true;
    const std::string ban2 = "rocm_smi:::clk_freq_level_2:device=" + std::to_string(dev);
    if (ev == ban2) return true;
    return false;
}

// parsing helpers (no regex)
static bool extract_event_token(const std::string& line, std::string& out) {
    size_t p = line.find("amd_smi:::");
    size_t p2 = line.find("rocm_smi:::");
    size_t pos = (p!=std::string::npos ? p : p2);
    if (pos == std::string::npos) return false;
    size_t i = pos;
    while (i < line.size() && !is_space(line[i]) && line[i] != '|') ++i;
    if (i == pos) return false;
    out = line.substr(pos, i-pos);
    return true;
}
static std::string normalize_device(const std::string& ev, int dev) {
    std::string out = ev;
    size_t dpos = out.find(":device=");
    if (dpos != std::string::npos) {
        size_t val = dpos + 8, end = val;
        while (end < out.size() && std::isdigit((unsigned char)out[end])) ++end;
        out.replace(val, end-val, std::to_string(dev));
    } else {
        out += ":device=" + std::to_string(dev);
    }
    return out;
}
static bool split_event(const std::string& ev, std::string& metric, int& device, int& sensor, bool& has_sensor) {
    size_t pfx;
    if      (starts_with(ev, "amd_smi:::"))  pfx = 10;
    else if (starts_with(ev, "rocm_smi:::")) pfx = 11;
    else return false;
    size_t dpos = ev.find(":device=", pfx);
    if (dpos == std::string::npos) return false;
    metric = ev.substr(pfx, dpos - pfx);
    size_t vpos = dpos + 8, vend = vpos;
    while (vend < ev.size() && std::isdigit((unsigned char)ev[vend])) ++vend;
    if (vend == vpos) return false;
    device = std::atoi(ev.substr(vpos, vend-vpos).c_str());
    has_sensor = false; sensor = 0;
    size_t spos = ev.find(":sensor=", vend);
    if (spos != std::string::npos) {
        size_t sv = spos + 8, se = sv;
        while (se < ev.size() && std::isdigit((unsigned char)ev[se])) ++se;
        if (se > sv) { sensor = std::atoi(ev.substr(sv, se-sv).c_str()); has_sensor = true; }
    }
    return true;
}
static std::vector<std::string> parse_events_file_all(const std::string& path, int dev) {
    std::vector<std::string> names;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return names;
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        std::string line(buf), tok;
        if (extract_event_token(line, tok)) {
            std::string ev = normalize_device(tok, dev);
            if (!should_drop_event_all(ev, dev)) names.push_back(ev);
        }
    }
    fclose(f);
    return names;
}

static bool classify_event(const std::string& ev, MetricDef& out) {
    if (ev.find("fan") != std::string::npos) return false; // skip fans
    std::string metric; int dev=0, sensor=0; bool has_sensor=false;
    if (!split_event(ev, metric, dev, sensor, has_sensor)) return false;
    out.name = ev; out.arg = std::max(0, std::min(7, sensor));

    amdsmi_temperature_metric_t tmet;
    if (temp_metric_from_string(metric, &tmet)) { out.kind = Kind::TEMP_METRIC; out.arg2 = (int)tmet; return true; }
    if      (metric=="gfx_activity") { out.kind = Kind::GFX_ACTIVITY; return true; }
    else if (metric=="umc_activity") { out.kind = Kind::UMC_ACTIVITY; return true; }
    else if (metric=="mm_activity")  { out.kind = Kind::MM_ACTIVITY;  return true; }
    if      (metric=="mem_total_VRAM"){ out.kind = Kind::VRAM_TOTAL; return true; }
    else if (metric=="mem_usage_VRAM"){ out.kind = Kind::VRAM_USED;  return true; }
    if      (metric=="power_average") { out.kind = Kind::POWER_AVG_W;  return true; }
    else if (metric=="power_cap_current")   { out.kind = Kind::POWER_CAP_CUR; return true; }
    else if (metric=="power_cap_range_min") { out.kind = Kind::POWER_CAP_MIN; return true; }
    else if (metric=="power_cap_range_max") { out.kind = Kind::POWER_CAP_MAX; return true; }
    if      (metric=="clk_freq_current") { out.kind = Kind::CLK_SYS_CURRENT_MHZ; return true; }
    else if (metric=="clk_freq_count")   { out.kind = Kind::CLK_SYS_COUNT;       return true; }
    else if (starts_with(metric, "clk_freq_level_")) { out.kind = Kind::CLK_SYS_LEVEL; out.arg2 = std::atoi(metric.c_str()+std::strlen("clk_freq_level_")); return true; }
    if      (metric=="pci_throughput_sent")      { out.kind = Kind::PCIE_THROUGHPUT_SENT;     return true; }
    else if (metric=="pci_throughput_received")  { out.kind = Kind::PCIE_THROUGHPUT_RECEIVED; return true; }
    else if (metric=="pci_throughput_max_packet"){ out.kind = Kind::PCIE_THROUGHPUT_MAXPKT;   return true; }
    else if (metric=="pci_replay_counter")       { out.kind = Kind::PCIE_REPLAY_COUNTER;      return true; }
    if (metric=="energy_consumed") { out.kind = Kind::ENERGY_CONSUMED; return true; }
    if (metric=="bdfid" || metric=="bdf_id") { out.kind = Kind::GPU_BDFID; return true; }
    if (metric=="numa_node") { out.kind = Kind::NUMA_NODE; return true; }
    if (metric=="gpu_id") { out.kind = Kind::GPU_ID; return true; }
    if (metric=="gpu_revision") { out.kind = Kind::GPU_REVISION; return true; }
    if (metric=="gpu_subsystem_id") { out.kind = Kind::GPU_SUBSYSTEM_ID; return true; }
    return false;
}

// ---------- PAPI timing ----------
static Stat time_papi_read_one_metric(const std::string& event, size_t iters, bool verify, std::string& notes) {
    int EventSet = PAPI_NULL; long long tmp = 0; int st;
    st = PAPI_create_eventset(&EventSet);
    if (st != PAPI_OK) { notes = PAPI_strerror(st); return Stat{}; }
    int evcode = PAPI_NULL;
    st = PAPI_event_name_to_code(event.c_str(), &evcode);
    if (st != PAPI_OK) { notes = PAPI_strerror(st); PAPI_cleanup_eventset(EventSet); PAPI_destroy_eventset(&EventSet); return Stat{}; }
    st = PAPI_add_event(EventSet, evcode);
    if (st != PAPI_OK) { notes = PAPI_strerror(st); PAPI_cleanup_eventset(EventSet); PAPI_destroy_eventset(&EventSet); return Stat{}; }
    st = PAPI_start(EventSet);
    if (st != PAPI_OK) { notes = PAPI_strerror(st); PAPI_cleanup_eventset(EventSet); PAPI_destroy_eventset(&EventSet); return Stat{}; }

    bool printed = false;
    Stat s; s.n = iters;
    for (int w=0; w<2 && iters>0; ++w) (void)PAPI_read(EventSet, &tmp);

    for (size_t i=0;i<iters;i++) {
        uint64_t t0 = now_ns();
        int rc = PAPI_read(EventSet, &tmp);
        uint64_t dt = now_ns() - t0;

        s.min_ns = std::min(s.min_ns, dt);
        s.max_ns = std::max(s.max_ns, dt);
        s.sum_ns += dt;

        if (rc == PAPI_OK) {
            s.ok_count++;
            if (verify && !printed) { printf("[VERIFY][PAPI] %s = %lld\n", event.c_str(), tmp); printed = true; }
            if (PAPI_reset(EventSet) != PAPI_OK) { PAPI_stop(EventSet, &tmp); PAPI_start(EventSet); }
        } else if (notes.empty()) notes = PAPI_strerror(rc);
    }

    PAPI_stop(EventSet, &tmp);
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    return s;
}

// ---------- AMD SMI timing ----------
static Stat time_smi_one_metric(amdsmi_processor_handle h, const MetricDef& m, size_t iters, bool verify, std::string& notes) {
    static const amdsmi_temperature_type_t* TMP = kTempSensorMap;

    // warmup
    for (int w=0; w<2 && iters>0; ++w) {
        switch (m.kind) {
            case Kind::TEMP_METRIC: { int64_t t=0; (void)amdsmi_get_temp_metric(h, TMP[m.arg], (amdsmi_temperature_metric_t)m.arg2, &t); } break;
            case Kind::GFX_ACTIVITY:
            case Kind::UMC_ACTIVITY:
            case Kind::MM_ACTIVITY: { amdsmi_engine_usage_t u{}; (void)amdsmi_get_gpu_activity(h, &u); } break;
            case Kind::VRAM_TOTAL:  { uint64_t x=0; (void)amdsmi_get_gpu_memory_total(h, AMDSMI_MEM_TYPE_VRAM, &x); } break;
            case Kind::VRAM_USED:   { uint64_t x=0; (void)amdsmi_get_gpu_memory_usage(h, AMDSMI_MEM_TYPE_VRAM, &x); } break;
            case Kind::POWER_AVG_W: { amdsmi_power_info_t p{}; (void)amdsmi_get_power_info(h, &p); } break;
            case Kind::POWER_CAP_CUR:
            case Kind::POWER_CAP_MIN:
            case Kind::POWER_CAP_MAX: { amdsmi_power_cap_info_t pc{}; (void)amdsmi_get_power_cap_info(h, 0, &pc); } break;
            case Kind::CLK_SYS_CURRENT_MHZ:
            case Kind::CLK_SYS_COUNT:
            case Kind::CLK_SYS_LEVEL: { amdsmi_frequencies_t f{}; (void)amdsmi_get_clk_freq(h, AMDSMI_CLK_TYPE_SYS, &f); } break;
            case Kind::PCIE_THROUGHPUT_SENT:
            case Kind::PCIE_THROUGHPUT_RECEIVED:
            case Kind::PCIE_THROUGHPUT_MAXPKT: { uint64_t a=0,b=0,c=0; (void)amdsmi_get_gpu_pci_throughput(h, &a,&b,&c); } break;
            case Kind::PCIE_REPLAY_COUNTER: { uint64_t c=0; (void)amdsmi_get_gpu_pci_replay_counter(h,&c); } break;
            case Kind::ENERGY_CONSUMED: { uint64_t e=0, ts=0; float res=0; (void)amdsmi_get_energy_count(h,&e,&res,&ts); } break;
            case Kind::GPU_BDFID: { uint64_t id=0; (void)amdsmi_get_gpu_bdf_id(h,&id); } break;
            case Kind::NUMA_NODE: { int32_t n=0; (void)amdsmi_get_gpu_topo_numa_affinity(h,&n); } break;
            case Kind::GPU_ID: { uint16_t v=0; (void)amdsmi_get_gpu_id(h,&v); } break;
            case Kind::GPU_REVISION: { uint16_t r=0; (void)amdsmi_get_gpu_revision(h,&r); } break;
            case Kind::GPU_SUBSYSTEM_ID: { uint16_t s=0; (void)amdsmi_get_gpu_subsystem_id(h,&s); } break;
        }
    }

    bool printed = false; Stat s; s.n = iters;
    for (size_t i=0;i<iters;i++) {
        uint64_t t0 = now_ns();
        amdsmi_status_t rc = AMDSMI_STATUS_SUCCESS;
        switch (m.kind) {
            case Kind::TEMP_METRIC: { int64_t t=0; rc = amdsmi_get_temp_metric(h, TMP[m.arg], (amdsmi_temperature_metric_t)m.arg2, &t);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS) { printf("[VERIFY][AMDSMI] %s = %.3f C\n", m.name.c_str(), (double)t/1000.0); printed = true; } } break;
            case Kind::GFX_ACTIVITY:
            case Kind::UMC_ACTIVITY:
            case Kind::MM_ACTIVITY: { amdsmi_engine_usage_t u{}; rc = amdsmi_get_gpu_activity(h, &u);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS) { uint32_t v=(m.kind==Kind::GFX_ACTIVITY)?u.gfx_activity:(m.kind==Kind::UMC_ACTIVITY)?u.umc_activity:u.mm_activity;
                    printf("[VERIFY][AMDSMI] %s = %u %%\n", m.name.c_str(), v); printed = true; } } break;
            case Kind::VRAM_TOTAL:  { uint64_t x=0; rc = amdsmi_get_gpu_memory_total(h, AMDSMI_MEM_TYPE_VRAM, &x);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS) { printf("[VERIFY][AMDSMI] %s = %llu bytes\n", m.name.c_str(), (unsigned long long)x); printed = true; } } break;
            case Kind::VRAM_USED:   { uint64_t x=0; rc = amdsmi_get_gpu_memory_usage(h, AMDSMI_MEM_TYPE_VRAM, &x);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS) { printf("[VERIFY][AMDSMI] %s = %llu bytes\n", m.name.c_str(), (unsigned long long)x); printed = true; } } break;
            case Kind::POWER_AVG_W: { amdsmi_power_info_t p{}; rc = amdsmi_get_power_info(h, &p);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS) { printf("[VERIFY][AMDSMI] %s = avg %.3f W\n", m.name.c_str(), (double)p.average_socket_power); printed = true; } } break;
            case Kind::POWER_CAP_CUR:
            case Kind::POWER_CAP_MIN:
            case Kind::POWER_CAP_MAX: { amdsmi_power_cap_info_t pc{}; rc = amdsmi_get_power_cap_info(h, 0, &pc);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS) { uint64_t v=(m.kind==Kind::POWER_CAP_CUR)?pc.power_cap:(m.kind==Kind::POWER_CAP_MIN)?pc.min_power_cap:pc.max_power_cap;
                    printf("[VERIFY][AMDSMI] %s = %llu W\n", m.name.c_str(), (unsigned long long)v); printed = true; } } break;
            case Kind::CLK_SYS_CURRENT_MHZ: { amdsmi_frequencies_t f{}; rc = amdsmi_get_clk_freq(h, AMDSMI_CLK_TYPE_SYS, &f);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS && f.num_supported>0 && f.current<f.num_supported) {
                    double mhz = (double)f.frequency[f.current]/1.0e6; printf("[VERIFY][AMDSMI] %s = %.2f MHz\n", m.name.c_str(), mhz); printed = true; } } break;
            case Kind::CLK_SYS_COUNT: { amdsmi_frequencies_t f{}; rc = amdsmi_get_clk_freq(h, AMDSMI_CLK_TYPE_SYS, &f);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS) { printf("[VERIFY][AMDSMI] %s = %u levels\n", m.name.c_str(), f.num_supported); printed = true; } } break;
            case Kind::CLK_SYS_LEVEL: { amdsmi_frequencies_t f{}; rc = amdsmi_get_clk_freq(h, AMDSMI_CLK_TYPE_SYS, &f);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS && (uint32_t)m.arg2<f.num_supported) {
                    double mhz=(double)f.frequency[m.arg2]/1.0e6; printf("[VERIFY][AMDSMI] %s = %.2f MHz\n", m.name.c_str(), mhz); printed = true; } } break;
            case Kind::PCIE_THROUGHPUT_SENT:
            case Kind::PCIE_THROUGHPUT_RECEIVED:
            case Kind::PCIE_THROUGHPUT_MAXPKT: { uint64_t a=0,b=0,c=0; rc = amdsmi_get_gpu_pci_throughput(h, &a,&b,&c); } break;
            case Kind::PCIE_REPLAY_COUNTER: { uint64_t c=0; rc = amdsmi_get_gpu_pci_replay_counter(h, &c); } break;
            case Kind::ENERGY_CONSUMED: { uint64_t e=0, ts=0; float res=0; rc = amdsmi_get_energy_count(h, &e, &res, &ts);
                if (verify && !printed && rc==AMDSMI_STATUS_SUCCESS) { printf("[VERIFY][AMDSMI] %s = %llu (res=%.6f)\n", m.name.c_str(), (unsigned long long)e, (double)res); printed = true; } } break;
            case Kind::GPU_BDFID: { uint64_t b=0; rc = amdsmi_get_gpu_bdf_id(h, &b); } break;
            case Kind::NUMA_NODE: { int32_t n=0; rc = amdsmi_get_gpu_topo_numa_affinity(h, &n); } break;
            case Kind::GPU_ID: { uint16_t v=0; rc = amdsmi_get_gpu_id(h, &v); } break;
            case Kind::GPU_REVISION: { uint16_t r=0; rc = amdsmi_get_gpu_revision(h, &r); } break;
            case Kind::GPU_SUBSYSTEM_ID: { uint16_t s=0; rc = amdsmi_get_gpu_subsystem_id(h, &s); } break;
        }
        uint64_t dt = now_ns() - t0;
        s.min_ns = std::min(s.min_ns, dt);
        s.max_ns = std::max(s.max_ns, dt);
        s.sum_ns += dt;
        if (rc == AMDSMI_STATUS_SUCCESS) s.ok_count++; else if (notes.empty()) notes = "non-OK AMDSMI status";
    }
    return s;
}

// ---------- default/key/pcie sets ----------
static std::vector<MetricDef> key_metrics_for_device(int dev) {
    char buf[256];
    auto N = [&](const char* fmt)->std::string{ std::snprintf(buf,sizeof(buf),fmt,dev); return std::string(buf); };
    std::vector<MetricDef> v;

    for (int s=0;s<8;s++) v.push_back({ N(("amd_smi:::temp_current:device=%d:sensor="+std::to_string(s)).c_str()), Kind::TEMP_METRIC, s, (int)AMDSMI_TEMP_CURRENT });

    v.push_back({ N("amd_smi:::gfx_activity:device=%d"), Kind::GFX_ACTIVITY, 0 });
    v.push_back({ N("amd_smi:::umc_activity:device=%d"), Kind::UMC_ACTIVITY, 0 });
    v.push_back({ N("amd_smi:::mm_activity:device=%d"),  Kind::MM_ACTIVITY,  0 });

    v.push_back({ N("amd_smi:::mem_total_VRAM:device=%d"), Kind::VRAM_TOTAL, 0 });
    v.push_back({ N("amd_smi:::mem_usage_VRAM:device=%d"), Kind::VRAM_USED,  0 });

    v.push_back({ N("amd_smi:::power_average:device=%d"), Kind::POWER_AVG_W, 0 });

    v.push_back({ N("amd_smi:::clk_freq_current:device=%d"), Kind::CLK_SYS_CURRENT_MHZ, 0 });
    v.push_back({ N("amd_smi:::clk_freq_count:device=%d"),   Kind::CLK_SYS_COUNT, 0 });

    return v;
}
static std::vector<MetricDef> pcie_metrics_for_device(int dev) {
    char buf[256];
    auto N = [&](const char* fmt)->std::string{ std::snprintf(buf,sizeof(buf),fmt,dev); return std::string(buf); };
    std::vector<MetricDef> v;
    v.push_back({ N("amd_smi:::pci_throughput_sent:device=%d"),      Kind::PCIE_THROUGHPUT_SENT, 0 });
    v.push_back({ N("amd_smi:::pci_throughput_received:device=%d"),  Kind::PCIE_THROUGHPUT_RECEIVED, 0 });
    v.push_back({ N("amd_smi:::pci_throughput_max_packet:device=%d"),Kind::PCIE_THROUGHPUT_MAXPKT, 0 });
    return v;
}

int main(int argc, char** argv) {
    Args a; if (!parse_args(argc, argv, a)) return 1;

    // Collect metrics according to set
    std::vector<MetricDef> metrics;
    if (a.which == Args::KEY) {
        metrics = key_metrics_for_device(a.device);
    } else if (a.which == Args::PCIE) {
        metrics = pcie_metrics_for_device(a.device);
    } else {
        // ALL: parse a.txt, drop PCIe throughput + unsupported clk_freq_level_2
        std::vector<std::string> names = parse_events_file_all(a.events_path, a.device);
        std::unordered_set<std::string> dedup;
        for (auto &ev : names) if (!dedup.count(ev)) {
            MetricDef m; if (classify_event(ev, m)) { metrics.push_back(m); dedup.insert(ev); }
        }
        if (metrics.empty()) {
            fprintf(stderr, "No events found in %s, falling back to 'key' set.\n", a.events_path.c_str());
            metrics = key_metrics_for_device(a.device);
        }
    }
    if (metrics.empty()) { fprintf(stderr, "No recognizable metrics.\n"); return 0; }

    // Initialize libraries (conditionally)
    if (!a.smi_only) {
        int st = PAPI_library_init(PAPI_VER_CURRENT);
        if (st != PAPI_VER_CURRENT && st > 0) { fprintf(stderr,"PAPI library mismatch (%d)\n", st); return 2; }
        if (st < 0) { fprintf(stderr,"PAPI init failed: %s\n", PAPI_strerror(st)); return 2; }
    }
    amdsmi_processor_handle h = nullptr;
    if (!a.papi_only) {
        if (amdsmi_init(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS) { fprintf(stderr,"amdsmi_init failed\n"); return 3; }
        h = find_gpu_by_logical_index(a.device);
        if (!h) { fprintf(stderr,"No GPU handle for device %d\n", a.device); amdsmi_shut_down(); return 3; }
    }

    FILE* csv = fopen(a.out.c_str(), "w"); if (!csv) { perror("open csv"); if(!a.papi_only) amdsmi_shut_down(); return 4; }
    write_csv_header(csv);

    for (const auto& m : metrics) {
        if (!a.smi_only) {
            std::string notes; Stat s = time_papi_read_one_metric(m.name, a.iters, a.verify, notes);
            write_csv_row(csv, "PAPI", m.name.c_str(), "PAPI_read", a.iters, s, notes);
        }
        if (!a.papi_only) {
            std::string notes; Stat s = time_smi_one_metric(h, m, a.iters, a.verify, notes);
            const char* api =
                (m.kind==Kind::TEMP_METRIC)              ? "amdsmi_get_temp_metric" :
                (m.kind==Kind::GFX_ACTIVITY ||
                 m.kind==Kind::UMC_ACTIVITY ||
                 m.kind==Kind::MM_ACTIVITY)             ? "amdsmi_get_gpu_activity" :
                (m.kind==Kind::VRAM_TOTAL)              ? "amdsmi_get_gpu_memory_total" :
                (m.kind==Kind::VRAM_USED)               ? "amdsmi_get_gpu_memory_usage" :
                (m.kind==Kind::POWER_AVG_W)             ? "amdsmi_get_power_info" :
                (m.kind==Kind::POWER_CAP_CUR ||
                 m.kind==Kind::POWER_CAP_MIN ||
                 m.kind==Kind::POWER_CAP_MAX)           ? "amdsmi_get_power_cap_info" :
                (m.kind==Kind::CLK_SYS_CURRENT_MHZ ||
                 m.kind==Kind::CLK_SYS_COUNT ||
                 m.kind==Kind::CLK_SYS_LEVEL)           ? "amdsmi_get_clk_freq" :
                (m.kind==Kind::PCIE_THROUGHPUT_SENT ||
                 m.kind==Kind::PCIE_THROUGHPUT_RECEIVED ||
                 m.kind==Kind::PCIE_THROUGHPUT_MAXPKT)  ? "amdsmi_get_gpu_pci_throughput" :
                (m.kind==Kind::PCIE_REPLAY_COUNTER)     ? "amdsmi_get_gpu_pci_replay_counter" :
                (m.kind==Kind::ENERGY_CONSUMED)         ? "amdsmi_get_energy_count" :
                (m.kind==Kind::GPU_BDFID)               ? "amdsmi_get_gpu_bdf_id" :
                (m.kind==Kind::NUMA_NODE)               ? "amdsmi_get_gpu_topo_numa_affinity" :
                (m.kind==Kind::GPU_ID)                  ? "amdsmi_get_gpu_id" :
                (m.kind==Kind::GPU_REVISION)            ? "amdsmi_get_gpu_revision" :
                (m.kind==Kind::GPU_SUBSYSTEM_ID)        ? "amdsmi_get_gpu_subsystem_id" : "amdsmi_*";
            write_csv_row(csv, "AMDSMI", m.name.c_str(), api, a.iters, s, notes);
        }
    }

    fclose(csv);
    if (!a.papi_only) amdsmi_shut_down();
    return 0;
}
