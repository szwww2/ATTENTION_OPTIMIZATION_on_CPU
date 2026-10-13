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
    t.fill_device_neutral(c.seed);
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
