// Tang's reproducible two-route scalar / SIMD / OpenMP timing experiment.
// Usage: attn_bench_tang L Bq warmups reps max_threads
// Example: attn_bench_tang 128 16 3 15 8
#include "attn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <omp.h>

namespace {

struct Options {
    std::size_t L;
    std::size_t bq;
    int warmups;
    int reps;
    int max_threads;
};

bool parse_positive(const char* arg, std::size_t& value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(arg, &end, 10);
    if (arg[0] == '-' || end == arg || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_options(int argc, char** argv, Options& o) {
    if (argc != 6) return false;
    std::size_t warmups = 0, reps = 0, threads = 0;
    if (!parse_positive(argv[1], o.L) || !parse_positive(argv[2], o.bq) ||
        !parse_positive(argv[3], warmups) || !parse_positive(argv[4], reps) ||
        !parse_positive(argv[5], threads) || warmups > 1000 || reps > 1000 ||
        threads > 256 || o.L > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    o.warmups = static_cast<int>(warmups);
    o.reps = static_cast<int>(reps);
    o.max_threads = static_cast<int>(threads);
    return true;
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    return n % 2 ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

template <typename Kernel>
bool measure(const char* route, const char* mode, const attn::Shape& s,
             const Options& o, int threads, std::size_t workspace_elems,
             const attn::Tensors& inputs, Kernel kernel) {
    // Workspace and output allocation are deliberately outside the timed region.
    std::vector<float> workspace(workspace_elems, 0.0f);
    std::vector<float> out(s.L * s.dv, 0.0f);
    for (int i = 0; i < o.warmups; ++i) kernel(workspace.data(), out.data());

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(o.reps));
    for (int i = 0; i < o.reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        kernel(workspace.data(), out.data());
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const attn::ErrorReport error =
        attn::check_contract(out.data(), inputs.o_ref.data(), out.size(), attn::Contract{});
    const double med = median(samples);
    const double min_ms = *std::min_element(samples.begin(), samples.end());
    std::printf("%s,%s,%llu,%llu,%llu,%llu,%d,%d,%d,%.6f,%.6f,%llu,%.9g,%.9g,%d,\"",
                route, mode, static_cast<unsigned long long>(s.L),
                static_cast<unsigned long long>(s.d), static_cast<unsigned long long>(s.dv),
                static_cast<unsigned long long>(o.bq), threads, o.warmups, o.reps,
                med, min_ms, static_cast<unsigned long long>(workspace_elems * sizeof(float)),
                error.max_abs, error.max_rel, error.pass ? 1 : 0);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) std::printf(";");
        std::printf("%.6f", samples[i]);
    }
    std::printf("\"\n");
    std::fflush(stdout);
    return error.pass;
}

}  // namespace

int main(int argc, char** argv) {
    Options o{};
    if (!parse_options(argc, argv, o)) {
        std::fprintf(stderr, "Usage: %s L Bq warmups reps max_threads (all positive)\n", argv[0]);
        return EXIT_FAILURE;
    }
    omp_set_dynamic(0);

    const attn::Shape s{o.L, 64, 64};
    attn::Tensors t;
    t.resize(s);
    t.fill_qkv(20261005ull + s.L);
    attn::oracle_fp64(s, t.q.data(), t.k.data(), t.v.data(), t.o_ref.data());

    std::printf("route,mode,L,d,dv,Bq,T,warmups,reps,median_ms,min_ms,workspace_bytes,max_abs,max_rel,contract_pass,samples_ms\n");
    bool okay = true;
    okay &= measure("full_matrix", "scalar", s, o, 1, attn::full_matrix_workspace_elems(s), t,
                    [&](float* ws, float* out) {
                        attn::full_matrix_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), ws, out);
                    });
    okay &= measure("full_matrix", "simd", s, o, 1, attn::full_matrix_workspace_elems(s), t,
                    [&](float* ws, float* out) {
                        attn::full_matrix_simd_fp32(s, t.q.data(), t.k.data(), t.v.data(), ws, out);
                    });
    okay &= measure("rowwise", "scalar", s, o, 1, attn::rowwise_workspace_elems(s, o.bq), t,
                    [&](float* ws, float* out) {
                        attn::rowwise_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), o.bq, ws, out);
                    });
    okay &= measure("rowwise", "simd", s, o, 1, attn::rowwise_workspace_elems(s, o.bq), t,
                    [&](float* ws, float* out) {
                        attn::rowwise_simd_fp32(s, t.q.data(), t.k.data(), t.v.data(), o.bq, ws, out);
                    });

    for (int threads = 1; threads <= o.max_threads; threads *= 2) {
        int observed_threads = 0;
#pragma omp parallel num_threads(threads)
        {
#pragma omp single
            observed_threads = omp_get_num_threads();
        }
        if (observed_threads != threads) {
            std::fprintf(stderr, "Requested %d OpenMP threads, observed %d.\n",
                         threads, observed_threads);
            return EXIT_FAILURE;
        }
        okay &= measure("full_matrix", "omp", s, o, threads,
                        attn::full_matrix_workspace_elems(s), t,
                        [&](float* ws, float* out) {
                            attn::full_matrix_simd_omp_fp32(s, t.q.data(), t.k.data(), t.v.data(),
                                                             o.bq, threads, ws, out);
                        });
        okay &= measure("rowwise", "omp", s, o, threads,
                        attn::rowwise_omp_workspace_elems(s, o.bq, threads), t,
                        [&](float* ws, float* out) {
                            attn::rowwise_simd_omp_fp32(s, t.q.data(), t.k.data(), t.v.data(),
                                                         o.bq, threads, ws, out);
                        });
        if (threads > o.max_threads / 2) break;
    }
    if (!okay) std::fprintf(stderr, "At least one route failed the FP64 contract.\n");
    return okay ? EXIT_SUCCESS : EXIT_FAILURE;
}
