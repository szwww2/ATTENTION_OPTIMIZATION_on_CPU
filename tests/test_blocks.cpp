// Block size and row order must not change the result: every split must pass and match bitwise.
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
    float q_scale;  // multiplies q after generation (stability stress when large)
};

const std::size_t kBlockSizes[] = {1, 0, 2, 3, 5, 7, 16, 33, 64, 128, 512};

// fixed-seed shuffle, so the order repeats
void shuffle_rows(std::vector<std::size_t>& v, std::uint64_t seed) {
    attn::SplitMix64 rng(seed);
    for (std::size_t i = v.size(); i > 1; --i) {
        const std::uint64_t j = rng.next_u64() % i;
        std::swap(v[i - 1], v[static_cast<std::size_t>(j)]);
    }
}

void run_case(const Case& c) {
    attn::Shape s{c.L, c.d, c.dv};
    attn::Tensors t;
    t.resize(s);
    t.fill_qkv(c.seed);
    if (c.q_scale != 1.0f) {
        for (float& x : t.q) x *= c.q_scale;
    }

    attn::oracle_fp64(s, t.q.data(), t.k.data(), t.v.data(), t.o_ref.data());

    const int failures_before = failures;
    std::vector<float> base;
    double worst_abs = 0.0;
    std::size_t blocks_tested = 0;

    for (std::size_t bq : kBlockSizes) {
        const std::size_t ws_elems = attn::rowwise_workspace_elems(s, bq);
        note(ws_elems == (bq == 0 ? 1 : bq) * s.L,
             std::string(c.name) + ": workspace matches effective Bq");
        std::vector<float> ws(ws_elems + 1, 12345.678f);
        std::vector<float> out(t.o.size(), 12345.678f);  // sentinel: must be overwritten
        attn::rowwise_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq, ws.data(), out.data());
        note(ws.back() == 12345.678f, std::string(c.name) + ": workspace guard unchanged");

        const attn::ErrorReport r =
            attn::check_contract(out.data(), t.o_ref.data(), out.size(), attn::Contract{});
        if (r.max_abs > worst_abs) worst_abs = r.max_abs;
        note(r.pass, std::string(c.name) + ": Bq=" + std::to_string(bq) + " passes the contract");
        ++blocks_tested;

        bool sentinel_left = false;
        for (float x : out) {
            if (x == 12345.678f) sentinel_left = true;
        }
        note(!sentinel_left,
             std::string(c.name) + ": Bq=" + std::to_string(bq) + " overwrites every output element");

        if (base.empty()) {
            base = out;
        } else {
            note(std::memcmp(base.data(), out.data(), out.size() * sizeof(float)) == 0,
                 std::string(c.name) + ": Bq=" + std::to_string(bq) + " bitwise equal to Bq=1");
        }
    }

    std::vector<std::size_t> natural(s.L);
    for (std::size_t i = 0; i < s.L; ++i) natural[i] = i;

    std::vector<std::size_t> reversed(natural.rbegin(), natural.rend());
    std::vector<std::size_t> shuffled = natural;
    shuffle_rows(shuffled, c.seed * 0x9E3779B97F4A7C15ull + 1u);

    std::vector<std::size_t> chunked;
    for (std::size_t begin = 0; begin < natural.size(); begin += 8) {
        const std::size_t end = std::min(begin + 8, natural.size());
        chunked.insert(chunked.end(), natural.begin() + static_cast<std::ptrdiff_t>(begin),
                       natural.begin() + static_cast<std::ptrdiff_t>(end));
    }
    std::reverse(chunked.begin(), chunked.end());

    std::vector<float> ws_one(attn::rowwise_workspace_elems(s, 1));

    struct Schedule {
        const char* name;
        const std::vector<std::size_t>* rows;
        bool one_row_per_call;
    };
    const Schedule schedules[] = {
        {"natural", &natural, false},
        {"reversed", &reversed, false},
        {"shuffled", &shuffled, false},
        {"chunks", &chunked, false},
        {"singletons", &shuffled, true},
    };
    std::size_t schedules_tested = 0;

    for (const Schedule& sc : schedules) {
        std::vector<float> out(t.o.size(), 12345.678f);
        if (sc.one_row_per_call) {
            for (std::size_t row : *sc.rows) {
                attn::rowwise_scalar_rows_fp32(s, t.q.data(), t.k.data(), t.v.data(), &row, 1,
                                               ws_one.data(), out.data());
            }
        } else {
            attn::rowwise_scalar_rows_fp32(s, t.q.data(), t.k.data(), t.v.data(), sc.rows->data(),
                                           sc.rows->size(), ws_one.data(), out.data());
        }
        ++schedules_tested;

        const attn::ErrorReport r =
            attn::check_contract(out.data(), t.o_ref.data(), out.size(), attn::Contract{});
        note(r.pass, std::string(c.name) + ": schedule '" + sc.name + "' passes the contract");
        note(std::memcmp(base.data(), out.data(), out.size() * sizeof(float)) == 0,
             std::string(c.name) + ": schedule '" + sc.name + "' bitwise equal to the natural order");
    }

    const std::size_t ws1 = attn::rowwise_workspace_elems(s, 1);
    const std::size_t ws_full = attn::full_matrix_workspace_elems(s);
    std::printf("  %-14s L=%-5llu d=%-3llu dv=%-3llu Bq x%2llu schedules x%llu max_abs=%.3e "
                "ws(Bq=1)=%-9llu L*L=%-9llu %s\n",
                c.name, static_cast<unsigned long long>(s.L), static_cast<unsigned long long>(s.d),
                static_cast<unsigned long long>(s.dv),
                static_cast<unsigned long long>(blocks_tested),
                static_cast<unsigned long long>(schedules_tested), worst_abs,
                static_cast<unsigned long long>(ws1), static_cast<unsigned long long>(ws_full),
                failures == failures_before ? "PASS" : "FAIL");
}

}  // namespace

int main() {
    std::printf("CEG5206 GP8 - W2 block-decomposition invariance suite (row-wise scalar route)\n");
    std::printf("checks: block-size invariance, row-schedule invariance, contract per decomposition\n\n");

    const Case cases[] = {
        {"single_row", 1, 64, 64, 1, 1.0f},
        {"odd_tails", 7, 31, 33, 1, 1.0f},
        {"non_power_two", 127, 64, 64, 1, 1.0f},
        {"odd_d", 129, 65, 64, 1, 1.0f},
        {"odd_dv", 256, 64, 33, 2, 1.0f},
        {"mid_grid", 512, 64, 64, 3, 1.0f},
        {"large_scores", 128, 64, 64, 1, 100.0f},
    };
    for (const Case& c : cases) run_case(c);

    // workspace comparison
    std::printf("\nworkspace required by the row-wise route (floats) vs the full-matrix route\n");
    std::printf("%8s %12s %12s %12s %10s\n", "L", "Bq=1", "Bq=64", "L*L", "L^2/(Bq=64)");
    const std::size_t sizes[] = {128, 512, 1024, 2048};
    for (std::size_t L : sizes) {
        attn::Shape s{L, 64, 64};
        const double small = static_cast<double>(attn::rowwise_workspace_elems(s, 64));
        const double full = static_cast<double>(attn::full_matrix_workspace_elems(s));
        std::printf("%8llu %12llu %12llu %12llu %9.2fx\n", static_cast<unsigned long long>(L),
                    static_cast<unsigned long long>(attn::rowwise_workspace_elems(s, 1)),
                    static_cast<unsigned long long>(attn::rowwise_workspace_elems(s, 64)),
                    static_cast<unsigned long long>(attn::full_matrix_workspace_elems(s)),
                    full / small);
    }

    std::printf("\n%d checks, %d failures -> %s\n", checks, failures,
                failures == 0 ? "BLOCK SUITE PASSED" : "BLOCK SUITE FAILED");
    return failures == 0 ? 0 : 1;
}
