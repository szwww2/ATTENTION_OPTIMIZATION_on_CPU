#include "attn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void note(bool ok, const std::string& label) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("  FAIL %s\n", label.c_str());
    }
}

struct Case {
    const char* name;
    std::size_t L, d, dv;
    std::uint64_t seed;
    float q_scale;
    bool zero_q;
    bool repeat;
};

void run_case(const Case& c) {
    const attn::Shape s{c.L, c.d, c.dv};
    attn::Tensors t;
    t.resize(s);
    t.fill_qkv(c.seed);
    if (c.zero_q) std::fill(t.q.begin(), t.q.end(), 0.0f);
    if (c.q_scale != 1.0f) {
        for (float& x : t.q) x *= c.q_scale;
    }
    attn::oracle_fp64(s, t.q.data(), t.k.data(), t.v.data(), t.o_ref.data());

    std::vector<float> full_reference(t.o.size());
    std::vector<float> full_ws(attn::full_matrix_workspace_elems(s));
    attn::full_matrix_simd_fp32(s, t.q.data(), t.k.data(), t.v.data(),
                                full_ws.data(), full_reference.data());
    std::vector<float> row_reference(t.o.size());
    std::vector<float> row_ws(attn::rowwise_workspace_elems(s, 1));
    attn::rowwise_simd_fp32(s, t.q.data(), t.k.data(), t.v.data(), 1,
                            row_ws.data(), row_reference.data());

    const int failures_before = failures;
    const int thread_counts[] = {1, 2, 4};
    const std::size_t block_sizes[] = {0, 1, 7, 512};
    constexpr float guard = -12345.0f;
    constexpr float output_sentinel = 12345.678f;

    for (int threads : thread_counts) {
        int observed_threads = 0;
#pragma omp parallel num_threads(threads)
        {
#pragma omp single
            observed_threads = omp_get_num_threads();
        }
        note(observed_threads == threads,
             std::string(c.name) + ": OpenMP created T=" + std::to_string(threads));

        for (std::size_t bq : block_sizes) {
            const std::string config = std::string(c.name) + " T=" + std::to_string(threads) +
                                       " Bq=" + std::to_string(bq);

            std::vector<float> full_scores(attn::full_matrix_workspace_elems(s) + 2, guard);
            std::vector<float> full_output(t.o.size(), output_sentinel);
            attn::full_matrix_simd_omp_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq,
                                             threads, full_scores.data() + 1, full_output.data());
            note(attn::check_contract(full_output.data(), t.o_ref.data(), full_output.size(),
                                      attn::Contract{}).pass,
                 config + ": full-matrix passes FP64 contract");
            note(std::memcmp(full_output.data(), full_reference.data(),
                             full_output.size() * sizeof(float)) == 0,
                 config + ": full-matrix matches single-thread SIMD bitwise");
            note(std::memcmp(full_scores.data() + 1, full_ws.data(),
                             full_ws.size() * sizeof(float)) == 0,
                 config + ": full-matrix P workspace matches serial SIMD bitwise");
            note(full_scores.front() == guard && full_scores.back() == guard,
                 config + ": full-matrix workspace guards unchanged");

            const std::size_t row_elems = attn::rowwise_omp_workspace_elems(s, bq, threads);
            std::vector<float> row_scores(row_elems + 2, guard);
            std::vector<float> row_output(t.o.size(), output_sentinel);
            attn::rowwise_simd_omp_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq,
                                         threads, row_scores.data() + 1, row_output.data());
            note(attn::check_contract(row_output.data(), t.o_ref.data(), row_output.size(),
                                      attn::Contract{}).pass,
                 config + ": row-wise passes FP64 contract");
            note(std::memcmp(row_output.data(), row_reference.data(),
                             row_output.size() * sizeof(float)) == 0,
                 config + ": row-wise matches single-thread SIMD bitwise");
            note(row_scores.front() == guard && row_scores.back() == guard,
                 config + ": row-wise workspace guards unchanged");

            if (c.repeat) {
                std::fill(full_output.begin(), full_output.end(), output_sentinel);
                std::fill(row_output.begin(), row_output.end(), output_sentinel);
                attn::full_matrix_simd_omp_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq,
                                                 threads, full_scores.data() + 1,
                                                 full_output.data());
                attn::rowwise_simd_omp_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq,
                                             threads, row_scores.data() + 1,
                                             row_output.data());
                note(std::memcmp(full_output.data(), full_reference.data(),
                                 full_output.size() * sizeof(float)) == 0,
                     config + ": full-matrix repeated call matches");
                note(std::memcmp(row_output.data(), row_reference.data(),
                                 row_output.size() * sizeof(float)) == 0,
                     config + ": row-wise repeated call matches");
            }
        }
    }

    std::printf("  %-16s L=%-4llu d=%-3llu dv=%-3llu %s\n", c.name,
                static_cast<unsigned long long>(c.L),
                static_cast<unsigned long long>(c.d),
                static_cast<unsigned long long>(c.dv),
                failures == failures_before ? "PASS" : "FAIL");
}

}  // namespace

int main() {
    omp_set_dynamic(0);
    note(attn::rowwise_workspace_elems(attn::Shape{10, 2, 2}, 0) == 10,
         "Bq=0 falls back to one row of scratch");
    note(attn::rowwise_omp_workspace_elems(attn::Shape{10, 2, 2}, 3, 2) == 60,
         "L=10 Bq=3 T=2 requires 60 scratch floats");

    const Case cases[] = {
        {"single_row", 1, 64, 64, 1, 1.0f, false, false},
        {"odd_tails", 7, 31, 33, 2, 1.0f, false, false},
        {"odd_d", 129, 65, 64, 3, 1.0f, false, false},
        {"odd_dv", 256, 64, 33, 4, 1.0f, false, false},
        {"equal_scores", 64, 64, 64, 5, 1.0f, true, false},
        {"large_scores", 128, 64, 64, 6, 100.0f, false, false},
        {"repeated_calls", 256, 64, 64, 7, 1.0f, false, true},
    };
    for (const Case& c : cases) run_case(c);

    std::printf("%d checks, %d failures -> %s\n", checks, failures,
                failures == 0 ? "OPENMP SUITE PASSED" : "OPENMP SUITE FAILED");
    return failures == 0 ? 0 : 1;
}
