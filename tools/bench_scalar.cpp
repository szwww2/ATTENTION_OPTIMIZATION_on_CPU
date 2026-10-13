// Timing: warm-up, repeated runs, median latency, GFLOP/s, workspace. --csv for CSV output.
#include "attn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

double median_ms(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    const bool write_csv = (argc > 1 && std::string(argv[1]) == "--csv");
    if (write_csv) {
        std::printf("L,d,dv,reps,median_ms,min_ms,gflops,workspace_MB,roundtrip_max_abs\n");
    } else {
        std::printf("CEG5206 GP8 - full-matrix scalar FP32 baseline\n");
        std::printf("%6s %4s %4s %5s %12s %12s %10s %12s\n", "L", "d", "dv", "reps", "median_ms",
                    "min_ms", "GFLOP/s", "ws_MB");
    }

    struct Point {
        std::size_t L;
        int warmup, reps;
    };
    const Point points[] = {
        {128, 3, 20}, {512, 3, 10}, {1024, 2, 5}, {2048, 2, 3},
    };

    for (const Point& p : points) {
        attn::Shape s{p.L, 64, 64};
        attn::Tensors t;
        t.resize(s);
        t.fill_device_neutral(/*seed=*/20261005ull + p.L);
        t.ws.assign(attn::full_matrix_workspace_elems(s), 0.0f);

        for (int i = 0; i < p.warmup; ++i) {
            attn::full_matrix_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), t.ws.data(),
                                          t.o.data());
        }

        std::vector<double> times;
        times.reserve(static_cast<std::size_t>(p.reps));
        for (int i = 0; i < p.reps; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            attn::full_matrix_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), t.ws.data(),
                                          t.o.data());
            const auto t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        // a timing row only means something if the result still passes
        attn::oracle_fp64(s, t.q.data(), t.k.data(), t.v.data(), t.o_ref.data());
        const attn::ErrorReport r =
            attn::check_contract(t.o.data(), t.o_ref.data(), t.o.size(), attn::Contract{});

        const double med = median_ms(times);
        const double best = *std::min_element(times.begin(), times.end());
        const double flops = 2.0 * static_cast<double>(s.L) * static_cast<double>(s.L) *
                             static_cast<double>(s.d + s.dv);
        const double gflops = flops / (med * 1e-3) / 1e9;
        const double ws_mb = static_cast<double>(t.ws.size()) * sizeof(float) / 1e6;

        if (write_csv) {
            std::printf("%llu,%llu,%llu,%d,%.6f,%.6f,%.4f,%.3f,%.3e\n",
                        static_cast<unsigned long long>(s.L), static_cast<unsigned long long>(s.d),
                        static_cast<unsigned long long>(s.dv), p.reps, med, best, gflops, ws_mb,
                        r.max_abs);
        } else {
            std::printf("%6llu %4llu %4llu %5d %12.3f %12.3f %10.3f %12.3f%s\n",
                        static_cast<unsigned long long>(s.L), static_cast<unsigned long long>(s.d),
                        static_cast<unsigned long long>(s.dv), p.reps, med, best, gflops, ws_mb,
                        r.pass ? "" : "  <-- CONTRACT FAILED");
        }
        if (!r.pass) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
