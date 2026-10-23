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

enum class Route { full_matrix, rowwise, streaming };

const char* route_name(Route r) {
    if (r == Route::rowwise) return "rowwise";
    if (r == Route::streaming) return "streaming";
    return "full_matrix";
}

void run_route(const attn::Shape& s, attn::Tensors& t, Route r, std::size_t bq, float* out) {
    if (r == Route::full_matrix) {
        attn::full_matrix_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), t.ws.data(), out);
    } else if (r == Route::rowwise) {
        attn::rowwise_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq, t.ws.data(), out);
    } else {
        attn::blocked_streaming_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq, bq,
                                            t.ws.data(), out);
    }
}

void run_case(const Case& c, Route route) {
    attn::Shape s{c.L, c.d, c.dv};
    attn::Tensors t;
    t.resize(s);
    const std::size_t bq = std::min<std::size_t>(64, s.L);
    if (route == Route::full_matrix) t.ws.assign(attn::full_matrix_workspace_elems(s), 0.0f);
    else if (route == Route::rowwise) t.ws.assign(attn::rowwise_workspace_elems(s, bq), 0.0f);
    else t.ws.assign(attn::blocked_streaming_workspace_elems(s, bq, bq), 0.0f);
    t.fill_qkv(c.seed);
    if (c.zero_q) std::fill(t.q.begin(), t.q.end(), 0.0f);
    if (c.q_scale != 1.0f) {
        for (float& x : t.q) x *= c.q_scale;
    }

    attn::oracle_fp64(s, t.q.data(), t.k.data(), t.v.data(), t.o_ref.data());
    run_route(s, t, route, bq, t.o.data());
    const attn::ErrorReport r =
        attn::check_contract(t.o.data(), t.o_ref.data(), t.o.size(), attn::Contract{});

    // %llu + explicit casts: portable across MinGW / MSVC / glibc printfs.
    std::printf("  %-10s %-16s L=%-5llu d=%-3llu dv=%-3llu seed=%-3llu max_abs=%.3e max_rel=%.3e %s\n",
                route_name(route), c.name, static_cast<unsigned long long>(c.L),
                static_cast<unsigned long long>(c.d), static_cast<unsigned long long>(c.dv),
                static_cast<unsigned long long>(c.seed), r.max_abs, r.max_rel,
                r.pass ? "PASS" : "FAIL");

    const std::string label = std::string(c.name) + " [" + route_name(route) + "]";
    note(r.non_finite == 0, label + ": all outputs finite");
    note(r.pass, label + ": contract |diff| <= 1e-4 + 1e-3|ref|");

    if (c.zero_q) {
        std::vector<double> uniform(t.o.size(), 0.0);
        attn::uniform_attention_reference(s, t.v.data(), uniform.data());
        const attn::ErrorReport ru =
            attn::check_contract(t.o.data(), uniform.data(), t.o.size(), attn::Contract{});
        note(ru.pass, label + ": matches the analytic uniform-attention reference");
    }

    if (c.repeat) {
        std::vector<float> second(t.o.size(), 12345.678f);  // sentinel: must be fully overwritten
        run_route(s, t, route, bq, second.data());
        note(std::memcmp(second.data(), t.o.data(), t.o.size() * sizeof(float)) == 0,
             label + ": repeated call is bitwise identical");
        bool sentinel_left = false;
        for (float x : second) {
            if (x == 12345.678f) sentinel_left = true;
        }
        note(!sentinel_left, label + ": output buffer fully overwritten");
    }
}

}  // namespace

int main() {
    std::printf("CEG5206 GP8 - W1 numeric contract suite\n");
    std::printf("rule: |O_hat - O64| <= 1e-4 + 1e-3 |O64|, relative denominator max(1e-8, |O64|)\n");

    // Required boundary / tail shapes, three recorded seeds each.
    const Case shapes[] = {
        {"single_row", 1, 64, 64, 1, 1.0f, false, false, "L = 1"},
        {"odd_tails", 7, 31, 33, 1, 1.0f, false, false, "non-multiple of tile and SIMD widths"},
        {"power_two", 128, 64, 64, 1, 1.0f, false, false, "grid point L = 128"},
        {"non_power_two", 127, 64, 64, 1, 1.0f, false, false, "L not a power of two"},
        {"odd_d", 129, 65, 64, 1, 1.0f, false, false, "d = 65"},
        {"odd_dv", 256, 64, 33, 1, 1.0f, false, false, "dv = 33"},
    };
    Case specials[] = {
        {"equal_scores", 64, 64, 64, 1, 1.0f, true, false, "q = 0: uniform softmax, analytic check"},
        {"large_scores", 128, 64, 64, 1, 100.0f, false, false,
         "score spread ~ +/-130: max subtraction must keep exp finite"},
        {"repeated_calls", 256, 64, 64, 1, 1.0f, false, true, "determinism + output overwrite"},
    };

    const Route routes[] = {Route::full_matrix, Route::rowwise, Route::streaming};
    for (Route route : routes) {
        std::printf("\nroute %s\n", route_name(route));
        for (const Case& c : shapes) {
            for (std::uint64_t seed = 1; seed <= 3; ++seed) {
                Case cc = c;
                cc.seed = seed;
                run_case(cc, route);
            }
        }
        std::printf("\n");
        for (const Case& c : specials) run_case(c, route);
    }

    std::printf("\n%d checks, %d failures -> %s\n", checks, failures,
                failures == 0 ? "SUITE PASSED" : "SUITE FAILED");
    return failures == 0 ? 0 : 1;
}
