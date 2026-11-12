// W1 validation suite: every FP32 route must pass the frozen contract against the
// independent FP64 oracle on the test vectors required by the proposal:
//   L = 1, non-multiple tile / SIMD lengths, equal scores, large finite score
//   differences, repeated calls (determinism and full output overwrite).
#include "attn.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void note(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("    [FAIL] %s\n", what.c_str());
    }
}

struct Case {
    const char* name;
    std::size_t L, d, dv;
    std::uint64_t seed;
    float q_scale;      // multiplies q after generation (stability stress when large)
    bool zero_q;        // q = 0 -> equal scores -> analytic reference
    bool repeat;        // run twice: bitwise determinism + output overwrite
    const char* why;
};

void run_case(const Case& c) {
    attn::Shape s{c.L, c.d, c.dv};
    attn::Tensors t;
    t.resize(s);
    t.ws.assign(attn::full_matrix_workspace_elems(s), 0.0f);
    t.fill_qkv(c.seed);
    if (c.zero_q) std::fill(t.q.begin(), t.q.end(), 0.0f);
    if (c.q_scale != 1.0f) {
        for (float& x : t.q) x *= c.q_scale;
    }

    attn::oracle_fp64(s, t.q.data(), t.k.data(), t.v.data(), t.o_ref.data());
    attn::full_matrix_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), t.ws.data(), t.o.data());
    const attn::ErrorReport r =
        attn::check_contract(t.o.data(), t.o_ref.data(), t.o.size(), attn::Contract{});

    // %llu + explicit casts: portable across MinGW / MSVC / glibc printfs.
    std::printf("  %-16s L=%-5llu d=%-3llu dv=%-3llu seed=%-3llu max_abs=%.3e max_rel=%.3e %s\n",
                c.name, static_cast<unsigned long long>(c.L), static_cast<unsigned long long>(c.d),
                static_cast<unsigned long long>(c.dv), static_cast<unsigned long long>(c.seed),
                r.max_abs, r.max_rel, r.pass ? "PASS" : "FAIL");

    note(r.non_finite == 0, std::string(c.name) + ": all outputs finite");
    note(r.pass, std::string(c.name) + ": contract |diff| <= 1e-4 + 1e-3|ref|");

    if (c.zero_q) {
        std::vector<double> uniform(t.o.size(), 0.0);
        attn::uniform_attention_reference(s, t.v.data(), uniform.data());
        const attn::ErrorReport ru =
            attn::check_contract(t.o.data(), uniform.data(), t.o.size(), attn::Contract{});
        note(ru.pass, std::string(c.name) + ": matches the analytic uniform-attention reference");
    }

    if (c.repeat) {
        std::vector<float> second(t.o.size(), 12345.678f);  // sentinel: must be fully overwritten
        attn::full_matrix_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), t.ws.data(), second.data());
        note(std::memcmp(second.data(), t.o.data(), t.o.size() * sizeof(float)) == 0,
             std::string(c.name) + ": repeated call is bitwise identical");
        bool sentinel_left = false;
        for (float x : second) {
            if (x == 12345.678f) sentinel_left = true;
        }
        note(!sentinel_left, std::string(c.name) + ": output buffer fully overwritten");
    }

#ifdef ATTN_TEST_SIMD
    auto validate_simd = [&](const std::string& route, const std::vector<float>& output) {
        const attn::ErrorReport sr =
            attn::check_contract(output.data(), t.o_ref.data(), output.size(), attn::Contract{});
        std::printf("    %-24s max_abs=%.3e max_rel=%.3e %s\n", route.c_str(),
                    sr.max_abs, sr.max_rel, sr.pass ? "PASS" : "FAIL");
        note(sr.non_finite == 0, std::string(c.name) + ": " + route + " outputs finite");
        note(sr.pass, std::string(c.name) + ": " + route + " passes FP64 contract");
        if (c.zero_q) {
            std::vector<double> uniform(output.size(), 0.0);
            attn::uniform_attention_reference(s, t.v.data(), uniform.data());
            note(attn::check_contract(output.data(), uniform.data(), output.size(),
                                      attn::Contract{}).pass,
                 std::string(c.name) + ": " + route + " matches uniform-attention reference");
        }
        if (c.repeat) {
            bool sentinel_left = false;
            for (float x : output) {
                if (x == 12345.678f) sentinel_left = true;
            }
            note(!sentinel_left, std::string(c.name) + ": " + route + " overwrites output");
        }
    };

    std::vector<float> full_simd_ws(attn::full_matrix_workspace_elems(s));
    std::vector<float> full_simd_out(t.o.size(), 12345.678f);
    attn::full_matrix_simd_fp32(s, t.q.data(), t.k.data(), t.v.data(),
                                full_simd_ws.data(), full_simd_out.data());
    validate_simd("full-matrix SIMD", full_simd_out);
    bool full_probs_normalized = true;
    for (std::size_t i = 0; i < s.L; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < s.L; ++j) {
            const float p = full_simd_ws[i * s.L + j];
            if (!std::isfinite(p) || p < 0.0f) full_probs_normalized = false;
            row_sum += p;
        }
        if (std::fabs(row_sum - 1.0) > 1e-4) full_probs_normalized = false;
    }
    note(full_probs_normalized, std::string(c.name) + ": full-matrix workspace contains P");
    if (c.repeat) {
        std::vector<float> second(t.o.size(), 12345.678f);
        attn::full_matrix_simd_fp32(s, t.q.data(), t.k.data(), t.v.data(),
                                    full_simd_ws.data(), second.data());
        note(std::memcmp(second.data(), full_simd_out.data(), second.size() * sizeof(float)) == 0,
             std::string(c.name) + ": full-matrix SIMD repeated call matches");
    }

    const std::size_t simd_blocks[] = {0, 1, 7, 16};
    std::vector<float> first_rowwise_simd;
    for (std::size_t bq : simd_blocks) {
        std::vector<float> row_simd_ws(attn::rowwise_workspace_elems(s, bq));
        std::vector<float> row_simd_out(t.o.size(), 12345.678f);
        attn::rowwise_simd_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq,
                                row_simd_ws.data(), row_simd_out.data());
        const std::string route = "row-wise SIMD Bq=" + std::to_string(bq);
        validate_simd(route, row_simd_out);
        if (first_rowwise_simd.empty()) {
            first_rowwise_simd = row_simd_out;
        } else {
            note(std::memcmp(first_rowwise_simd.data(), row_simd_out.data(),
                             row_simd_out.size() * sizeof(float)) == 0,
                 std::string(c.name) + ": " + route + " matches Bq=1 bitwise");
        }
        if (c.repeat) {
            std::vector<float> second(t.o.size(), 12345.678f);
            attn::rowwise_simd_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq,
                                    row_simd_ws.data(), second.data());
            note(std::memcmp(second.data(), row_simd_out.data(), second.size() * sizeof(float)) == 0,
                 std::string(c.name) + ": " + route + " repeated call matches");
        }
    }
#endif
}

}  // namespace

int main() {
    std::printf("CEG5206 GP8 - W1 numeric contract suite\n");
    std::printf("rule: |O_hat - O64| <= 1e-4 + 1e-3 |O64|, relative denominator max(1e-8, |O64|)\n\n");

    // Required boundary / tail shapes, three recorded seeds each.
    const Case shapes[] = {
        {"single_row", 1, 64, 64, 1, 1.0f, false, false, "L = 1"},
        {"odd_tails", 7, 31, 33, 1, 1.0f, false, false, "non-multiple of tile and SIMD widths"},
        {"power_two", 128, 64, 64, 1, 1.0f, false, false, "grid point L = 128"},
        {"non_power_two", 127, 64, 64, 1, 1.0f, false, false, "L not a power of two"},
        {"odd_d", 129, 65, 64, 1, 1.0f, false, false, "d = 65"},
        {"odd_dv", 256, 64, 33, 1, 1.0f, false, false, "dv = 33"},
    };
    for (const Case& c : shapes) {
        for (std::uint64_t seed = 1; seed <= 3; ++seed) {
            Case cc = c;
            cc.seed = seed;
            run_case(cc);
        }
    }

    std::printf("\n");
    Case specials[] = {
        {"equal_scores", 64, 64, 64, 1, 1.0f, true, false, "q = 0: uniform softmax, analytic check"},
        {"large_scores", 128, 64, 64, 1, 100.0f, false, false,
         "score spread ~ +/-130: max subtraction must keep exp finite"},
        {"repeated_calls", 256, 64, 64, 1, 1.0f, false, true, "determinism + output overwrite"},
    };
    for (const Case& c : specials) run_case(c);

    std::printf("\n%d checks, %d failures -> %s\n", checks, failures,
                failures == 0 ? "SUITE PASSED" : "SUITE FAILED");
    return failures == 0 ? 0 : 1;
}
