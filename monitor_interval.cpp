// monitor_interval.cpp
//
// Purpose:
//   Sweep polling intervals (e.g., 1,5,10,50,100 ms) for AMD-SMI metrics via PAPI,
//   and report per-interval (a) read overhead, and (b) how often each metric shows
//   new data (novelty) and unique-values-per-second.
//
//   Optional tight-loop mode for a single metric estimates intrinsic update cadence
//   by recording change-to-change time deltas with no sleeping.
//
// Build (Linux):
//   g++ -O2 -std=gnu++17 -Wall -Wextra -lpapi -lpthread -o monitor_interval monitor_interval.cpp
//
// Typical use:
//   # auto-pick first 8 usable AMD-SMI events; 10s per interval; write two summaries
//   ./monitor_interval --count 8 --intervals 1,5,10,50,100 --duration 10 \
//       --summary-poll poll_summary.csv --summary-metric metric_summary.csv
//
//   # supply your own event list (one event name per line) and also write time series
//   ./monitor_interval --events events.txt --intervals 2,10,100 --duration 8 \
//       --series-out series.csv
//
//   # tight-loop estimate for a single event (5 seconds), change threshold = 1
//   ./monitor_interval --tight "amd_smi:::power_average:device=0" \
//       --tight-seconds 5 --epsilon 1 --updates-out updates.csv
//
// Notes:
//   - Metrics are read via PAPI's amd_smi component only (no direct AMD SMI calls).
//   - Change detection uses integer deltas (abs diff >= epsilon), default epsilon=1.
//   - Read latency is the wall time of PAPI_read for the whole EventSet.
//   - CPU time per interval is measured via CLOCK_THREAD_CPUTIME_ID (Linux).
//
// Copyright:
//   This program follows the structure and conventions used in the user's prior
//   timing/CSV tools and PAPI-based AMD-SMI sampling harness. See thesis repo.
// ------------------------------------------------------------------------------

#include <papi.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
  #include <time.h>
#endif

// ---------- tiny utils ----------
static inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

static int extract_device_id(const std::string& name) {
    size_t pos = name.find(":device=");
    if (pos == std::string::npos) return -1;
    pos += 8;
    size_t end = pos;
    while (end < name.size() && std::isdigit((unsigned char)name[end])) ++end;
    if (end == pos) return -1;
    return std::atoi(name.substr(pos, end - pos).c_str());
}

static int infer_device_id(const std::vector<std::string>& names) {
    for (const auto& nm : names) {
        int dev = extract_device_id(nm);
        if (dev >= 0) return dev;
    }
    return -1;
}

static void die(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::exit(1);
}

static std::vector<int> parse_int_list_ms(const std::string& csv) {
    std::vector<int> out;
    std::string tok;
    for (size_t i=0;i<=csv.size();++i) {
        if (i==csv.size() || csv[i]==',' ) {
            if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
            tok.clear();
        } else if (!std::isspace((unsigned char)csv[i])) {
            tok.push_back(csv[i]);
        }
    }
    // drop non-positive intervals
    out.erase(std::remove_if(out.begin(), out.end(), [](int x){return x<=0;}), out.end());
    return out;
}

#if defined(__linux__)
static inline double thread_cpu_ms() {
    struct timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0.0;
    return ts.tv_sec*1000.0 + ts.tv_nsec/1.0e6;
}
#else
static inline double thread_cpu_ms() { return 0.0; } // best-effort on non-Linux
#endif

static double percentile_from_samples(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    if (p <= 0.0) return *std::min_element(v.begin(), v.end());
    if (p >= 100.0) return *std::max_element(v.begin(), v.end());
    double rank = (p/100.0) * (v.size()-1);
    size_t lo = (size_t)rank, hi = std::min(v.size()-1, lo+1);
    std::nth_element(v.begin(), v.begin()+lo, v.end());
    double v_lo = v[lo];
    if (hi==lo) return v_lo;
    std::nth_element(v.begin()+lo+1, v.begin()+hi, v.end());
    double frac = rank - lo;
    return v_lo + frac * (v[hi] - v_lo);
}

// ---------- CLI ----------
struct CLI {
    // event selection
    std::string events_path;  // optional: file with one PAPI event per line
    int count = 8;            // if events_path not provided, enumerate first N usable
    // sweep
    std::vector<int> intervals_ms{1,5,10,50,100};
    int duration_s = 10;
    int epsilon = 1;          // integer threshold for change detection
    // outputs
    std::string summary_poll = "poll_summary.csv";
    std::string summary_metric = "metric_summary.csv";
    std::string series_out;   // optional timeseries
    // tight-loop (single-event) mode
    std::string tight_event;
    int tight_seconds = 0;
    std::string updates_out = "updates.csv";
    // misc
    bool verbose = false;
};

static void usage_and_exit(const char* prog) {
    std::fprintf(stderr,
R"(Usage:
  %s [--events FILE | --count N] [--intervals CSV] [--duration S]
     [--epsilon INT] [--summary-poll PATH] [--summary-metric PATH]
     [--series-out PATH]
     [--tight EVENTNAME --tight-seconds S --updates-out PATH]
     [--verbose]

Examples:
  %s --count 8 --intervals 1,5,10,50,100 --duration 10
  %s --events my_events.txt --intervals 2,10,100 --duration 8 --series-out series.csv
  %s --tight "amd_smi:::power_average:device=0" --tight-seconds 5 --epsilon 1
)",
        prog, prog, prog, prog);
    std::exit(1);
}

static CLI parse_cli(int argc, char** argv) {
    CLI a;
    bool intervals_explicit = false;
    for (int i=1;i<argc;i++) {
        std::string s(argv[i]);
        auto need = [&](const char* flag){ if (i+1>=argc) die("Missing value for %s", flag); return std::string(argv[++i]); };
        if      (s=="--events")           a.events_path = need("--events");
        else if (s=="--count")            a.count = std::atoi(need("--count").c_str());
        else if (s=="--intervals")        { a.intervals_ms = parse_int_list_ms(need("--intervals")); intervals_explicit = true; }
        else if (s=="--duration")         a.duration_s = std::atoi(need("--duration").c_str());
        else if (s=="--epsilon")          a.epsilon = std::atoi(need("--epsilon").c_str());
        else if (s=="--summary-poll")     a.summary_poll = need("--summary-poll");
        else if (s=="--summary-metric")   a.summary_metric = need("--summary-metric");
        else if (s=="--series-out")       a.series_out = need("--series-out");
        else if (s=="--tight")            a.tight_event = need("--tight");
        else if (s=="--tight-seconds")    a.tight_seconds = std::atoi(need("--tight-seconds").c_str());
        else if (s=="--updates-out")      a.updates_out = need("--updates-out");
        else if (s=="--verbose")          a.verbose = true;
        else if (s=="--help" || s=="-h")  usage_and_exit(argv[0]);
        else die("Unknown arg: %s", s.c_str());
    }
    if (!intervals_explicit && !a.tight_event.empty()) a.intervals_ms.clear();
    if (a.intervals_ms.empty() && a.tight_event.empty()) die("No valid intervals; use --intervals CSV with positive ms.");
    if (a.count <= 0 && a.events_path.empty()) die("--count must be >0 (or provide --events).");
    if (!a.tight_event.empty() && a.tight_seconds <= 0)
        die("--tight given but --tight-seconds not set (positive).");
    return a;
}

// ---------- event selection (via PAPI amd_smi component) ----------

static int find_amd_smi_component() {
    int cid = -1;
    int ncomps = PAPI_num_components();
    for (int i=0;i<ncomps && cid<0; ++i) {
        const PAPI_component_info_t* c = PAPI_get_component_info(i);
        if (c && std::strcmp(c->name, "amd_smi")==0) cid = i;
    }
    return cid;
}

static std::vector<std::string> read_events_file(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream in(path);
    if (!in) die("Failed to open events file: %s", path.c_str());
    std::string line;
    while (std::getline(in, line)) {
        // trim
        size_t a=0, b=line.size();
        while (a<b && std::isspace((unsigned char)line[a])) ++a;
        while (b>a && std::isspace((unsigned char)line[b-1])) --b;
        if (a>=b) continue;
        if (line[a]=='#') continue;
        names.emplace_back(line.substr(a,b-a));
    }
    if (names.empty()) die("No events parsed from %s", path.c_str());
    return names;
}

static std::vector<std::string> enumerate_amd_smi_events(int cid, int want) {
    std::vector<std::string> names;
    int code = PAPI_NATIVE_MASK;
    if (PAPI_enum_cmp_event(&code, PAPI_ENUM_FIRST, cid) != PAPI_OK) return names;
    do {
        char name[PAPI_MAX_STR_LEN] = {0};
        if (PAPI_event_code_to_name(code, name) == PAPI_OK && name[0]) {
            // Skip process* events (not generally usable in this harness)
            if (std::strncmp(name, "amd_smi:::process", 17) == 0) {
                // skip
            } else {
                names.emplace_back(name);
            }
        }
        if ((int)names.size() >= want) break;
    } while (PAPI_enum_cmp_event(&code, PAPI_ENUM_EVENTS, cid) == PAPI_OK);
    return names;
}

static std::vector<int> add_events_or_die(int& EventSet, const std::vector<std::string>& names) {
    std::vector<int> codes; codes.reserve(names.size());
    for (const auto& nm : names) {
        int code = 0;
        int rc = PAPI_event_name_to_code(nm.c_str(), &code);
        if (rc != PAPI_OK) die("PAPI_event_name_to_code failed for %s: %s", nm.c_str(), PAPI_strerror(rc));
        rc = PAPI_add_event(EventSet, code);
        if (rc != PAPI_OK) die("PAPI_add_event failed for %s: %s", nm.c_str(), PAPI_strerror(rc));
        codes.push_back(code);
    }
    return codes;
}

// ---------- monitoring logic ----------

struct MetricStats {
    // per metric counts within an interval
    uint64_t reads = 0;
    uint64_t changes = 0;
    uint64_t unique_values = 0;
    long long last_value = 0;
    bool have_last = false;
    long long first_value = 0, final_value = 0;
    void observe(long long v, int epsilon) {
        ++reads;
        if (!have_last) {
            have_last = true;
            last_value = v;
            first_value = v;
            unique_values = 1;
        } else {
            if (std::llabs(v - last_value) >= epsilon) {
                ++changes;
                ++unique_values;
                last_value = v;
            }
        }
        final_value = v;
    }
};

struct PollSummary {
    int interval_ms = 0;
    int duration_s  = 0;
    uint64_t samples = 0;
    std::vector<double> read_us;   // per-sample read latencies (us)
    double cpu_ms = 0.0;           // thread CPU time over the interval
};

static void write_poll_summary_csv(const std::string& path,
                                   const std::vector<PollSummary>& rows) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) die("Cannot open %s", path.c_str());
    std::fprintf(f, "interval_ms,duration_s,samples,mean_us,p50_us,p95_us,p99_us,cpu_ms\n");
    for (auto& r : rows) {
        std::vector<double> tmp = r.read_us;
        double mean = tmp.empty()?0.0: std::accumulate(tmp.begin(), tmp.end(), 0.0)/tmp.size();
        double p50 = percentile_from_samples(tmp, 50.0);
        double p95 = percentile_from_samples(tmp, 95.0);
        double p99 = percentile_from_samples(tmp, 99.0);
        std::fprintf(f, "%d,%d,%" PRIu64 ",%.3f,%.3f,%.3f,%.3f,%.3f\n",
                     r.interval_ms, r.duration_s, r.samples, mean, p50, p95, p99, r.cpu_ms);
    }
    std::fclose(f);
}

static void write_metric_summary_csv(const std::string& path,
                                     const std::vector<std::string>& names,
                                     const std::vector<int>& intervals_ms,
                                     const std::vector<std::vector<MetricStats>>& all) {
    // all[interval_index][metric_index]
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) die("Cannot open %s", path.c_str());
    std::fprintf(f, "metric,interval_ms,duration_s,reads,changes,novelty_ratio,unique_values,unique_per_s,first,last\n");
    for (size_t i=0;i<intervals_ms.size();++i) {
        for (size_t m=0;m<names.size();++m) {
            const auto& st = all[i][m];
            double novelty = (st.reads ? (double)st.changes / (double)st.reads : 0.0);
            double uniq_per_s = 0.0;
            // duration_s = reads * (interval_ms/1000) approximately, but use exact from interval
            // The caller knows duration; we pass it as repeats * period. Here assume constant: reads≈samples
            // We'll infer duration from reads * interval_ms to keep CSV self-contained.
            double duration_s = (double)(st.reads) * ((double)intervals_ms[i]/1000.0);
            if (duration_s > 0.0) uniq_per_s = (double)st.unique_values / duration_s;
            std::fprintf(f, "\"%s\",%d,%.3f,%" PRIu64 ",%" PRIu64 ",%.6f,%" PRIu64 ",%.6f,%lld,%lld\n",
                         names[m].c_str(), intervals_ms[i], duration_s,
                         st.reads, st.changes, novelty, st.unique_values, uniq_per_s,
                         st.first_value, st.final_value);
        }
    }
    std::fclose(f);
}

static FILE* open_series_csv(const std::string& path, const std::vector<std::string>& names) {
    if (path.empty()) return nullptr;
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) die("Cannot open %s", path.c_str());
    std::fprintf(f, "t_sec");
    for (const auto& n : names) std::fprintf(f, ",\"%s\"", n.c_str());
    std::fprintf(f, "\n");
    return f;
}

static void write_series_row(FILE* f, double t_sec, const std::vector<long long>& vals) {
    if (!f) return;
    std::fprintf(f, "%.6f", t_sec);
    for (auto v : vals) std::fprintf(f, ",%lld", v);
    std::fprintf(f, "\n");
}

// Run a single interval
static void run_interval(int EventSet,
                         const std::vector<std::string>& names,
                         int interval_ms,
                         int duration_s,
                         int epsilon,
                         std::vector<MetricStats>& stats_out,
                         PollSummary& poll_out,
                         FILE* series,
                         int process_count_index) {
    const auto period = std::chrono::milliseconds(interval_ms);
    const auto t_start = std::chrono::steady_clock::now();
    const auto t_end   = t_start + std::chrono::seconds(duration_s);
    auto next_tick = t_start;

    stats_out.assign(names.size(), MetricStats{});
    poll_out.interval_ms = interval_ms;
    poll_out.duration_s  = duration_s;
    poll_out.read_us.clear();

    std::vector<long long> values(names.size(), 0);
    // warmup
    (void)PAPI_read(EventSet, values.data());

    double cpu0 = thread_cpu_ms();
    while (std::chrono::steady_clock::now() < t_end) {
        const uint64_t t0 = now_ns();
        int rc = PAPI_read(EventSet, values.data());
        const uint64_t t1 = now_ns();
        if (rc != PAPI_OK) die("PAPI_read failed: %s", PAPI_strerror(rc));
        double us = (t1 - t0) / 1000.0;
        poll_out.read_us.push_back(us);
        ++poll_out.samples;

        // per-metric novelty/update tracking
        for (size_t i=0;i<names.size();++i) {
            stats_out[i].observe(values[i], epsilon);
        }

        if (process_count_index >= 0 && process_count_index < (int)values.size()) {
            long long proc_cnt = values[process_count_index];
            if (proc_cnt > 1) {
                die("Process contention detected: process_count=%lld (expected <=1)", proc_cnt);
            }
        }

        if (series) {
            using namespace std::chrono;
            double tsec = duration_cast<std::chrono::duration<double>>(steady_clock::now() - t_start).count();
            write_series_row(series, tsec, values);
        }

        // pacing
        next_tick += period;
        auto now = std::chrono::steady_clock::now();
        if (next_tick > now) {
            std::this_thread::sleep_until(next_tick);
        } else {
            // running behind; skip sleeping
            next_tick = now;
        }
    }
    double cpu1 = thread_cpu_ms();
    poll_out.cpu_ms = (cpu1 - cpu0);
}

// ---------- tight-loop update-cadence probe (single event) ----------

static void run_tight_loop_one(const std::string& event_name,
                               int seconds,
                               int epsilon,
                               const std::string& out_path) {
    int rc = PAPI_library_init(PAPI_VER_CURRENT);
    if (rc != PAPI_VER_CURRENT) die("PAPI init failed: %s", PAPI_strerror(rc));
    int EventSet = PAPI_NULL;
    if ((rc = PAPI_create_eventset(&EventSet)) != PAPI_OK) die("PAPI_create_eventset: %s", PAPI_strerror(rc));
    int code = 0;
    if ((rc = PAPI_event_name_to_code(event_name.c_str(), &code)) != PAPI_OK) die("event_name_to_code(%s): %s", event_name.c_str(), PAPI_strerror(rc));
    if ((rc = PAPI_add_event(EventSet, code)) != PAPI_OK) die("PAPI_add_event(%s): %s", event_name.c_str(), PAPI_strerror(rc));
    if ((rc = PAPI_start(EventSet)) != PAPI_OK) die("PAPI_start: %s", PAPI_strerror(rc));

    const auto t0 = std::chrono::steady_clock::now();
    const auto t_end = t0 + std::chrono::seconds(seconds);
    long long v_prev = 0;
    bool have = false;

    std::vector<double> deltas_us;
    uint64_t last_change_ns = now_ns();

    while (std::chrono::steady_clock::now() < t_end) {
        long long v = 0;
        const uint64_t t_read0 = now_ns();
        rc = PAPI_read(EventSet, &v);
        (void)t_read0; // (we're not using read latency here)
        if (rc != PAPI_OK) die("PAPI_read: %s", PAPI_strerror(rc));
        if (!have) { have = true; v_prev = v; last_change_ns = now_ns(); continue; }
        if (std::llabs(v - v_prev) >= epsilon) {
            uint64_t now = now_ns();
            double du = (now - last_change_ns) / 1000.0;
            deltas_us.push_back(du);
            v_prev = v;
            last_change_ns = now;
        }
        // no sleep: tight loop
    }

    (void)PAPI_stop(EventSet, &v_prev);
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    FILE* f = std::fopen(out_path.c_str(), "w");
    if (!f) die("open %s failed", out_path.c_str());
    std::fprintf(f, "event,delta_us\n");
    for (double du : deltas_us) std::fprintf(f, "\"%s\",%.3f\n", event_name.c_str(), du);
    std::fclose(f);

    // print a small summary to stdout
    if (!deltas_us.empty()) {
        std::vector<double> tmp = deltas_us;
        double med = percentile_from_samples(tmp, 50.0);
        double p95 = percentile_from_samples(tmp, 95.0);
        std::fprintf(stdout, "[tight] %s  n=%zu  median=%.3f us  p95=%.3f us\n",
                     event_name.c_str(), deltas_us.size(), med, p95);
    } else {
        std::fprintf(stdout, "[tight] %s produced no value changes under epsilon=%d in %d s\n",
                     event_name.c_str(), epsilon, seconds);
    }
}

// ---------- main ----------

int main(int argc, char** argv) {
    CLI a = parse_cli(argc, argv);

    // Run only tight-loop mode if requested and nothing else
    if (!a.tight_event.empty() && a.intervals_ms.empty()) {
        run_tight_loop_one(a.tight_event, a.tight_seconds, a.epsilon, a.updates_out);
        return 0;
    }

    // Initialize PAPI once for sweep
    int rc = PAPI_library_init(PAPI_VER_CURRENT);
    if (rc != PAPI_VER_CURRENT) die("PAPI library init failed: %s", PAPI_strerror(rc));

    // Resolve events
    std::vector<std::string> event_names;
    if (!a.events_path.empty()) {
        event_names = read_events_file(a.events_path);
    } else {
        int cid = find_amd_smi_component();
        if (cid < 0) die("Unable to locate amd_smi component; is PAPI built with ROCm?");
        event_names = enumerate_amd_smi_events(cid, a.count);
        if ((int)event_names.size() < a.count) {
            std::fprintf(stderr, "Warning: requested %d events, got %zu usable.\n", a.count, event_names.size());
        }
        if (event_names.empty()) die("No usable amd_smi events found for enumeration.");
    }
    if (a.verbose) {
        std::fprintf(stdout, "Monitoring events (%zu):\n", event_names.size());
        for (size_t i=0;i<event_names.size();++i)
            std::fprintf(stdout, "  %2zu) %s\n", i+1, event_names[i].c_str());
    }

    // Create and start EventSet
    int EventSet = PAPI_NULL;
    if ((rc = PAPI_create_eventset(&EventSet)) != PAPI_OK) die("create_eventset: %s", PAPI_strerror(rc));

    // Ensure process_count is present for contention monitoring
    int dev_id = infer_device_id(event_names);
    if (dev_id < 0) dev_id = 0;
    std::string process_event = "amd_smi:::process_count:device=" + std::to_string(dev_id);
    int process_index = -1;
    for (size_t i=0;i<event_names.size();++i) {
        if (event_names[i].rfind("amd_smi:::process_count", 0) == 0) {
            process_index = (int)i;
            process_event = event_names[i];
            break;
        }
    }
    if (process_index < 0) {
        event_names.push_back(process_event);
        if (a.verbose) {
            std::fprintf(stdout, "Added %s for contention monitoring.\n", process_event.c_str());
        }
        process_index = (int)(event_names.size() - 1);
    }

    std::vector<int> codes = add_events_or_die(EventSet, event_names);
    (void)codes;
    if ((rc = PAPI_start(EventSet)) != PAPI_OK) die("PAPI_start: %s", PAPI_strerror(rc));

    // Prepare outputs
    FILE* series = open_series_csv(a.series_out, event_names);
    std::vector<PollSummary> poll_rows;
    std::vector<std::vector<MetricStats>> all_stats; // [interval][metric]

    // Sweep each interval
    for (int interval_ms : a.intervals_ms) {
        std::vector<MetricStats> stats;
        PollSummary row{};
        run_interval(EventSet, event_names, interval_ms, a.duration_s, a.epsilon, stats, row, series, process_index);
        poll_rows.push_back(std::move(row));
        all_stats.push_back(std::move(stats));
    }

    if (series) std::fclose(series);

    // Tear down PAPI
    std::vector<long long> final_readings(event_names.size(), 0);
    long long* final_ptr = final_readings.empty() ? nullptr : final_readings.data();
    (void)PAPI_stop(EventSet, final_ptr);
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    // Write summaries
    write_poll_summary_csv(a.summary_poll, poll_rows);
    write_metric_summary_csv(a.summary_metric, event_names, a.intervals_ms, all_stats);

    // Optional tight-loop pass if requested alongside sweep
    if (!a.tight_event.empty() && a.tight_seconds > 0) {
        run_tight_loop_one(a.tight_event, a.tight_seconds, a.epsilon, a.updates_out);
    }

    return 0;
}
