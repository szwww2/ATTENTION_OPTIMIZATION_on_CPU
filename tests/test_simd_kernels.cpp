#include "attn_simd.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void check(const char* name, const float* a, const float* b, std::size_t n,
           double expected, double tolerance) {
    const float got = attn::dot_avx2_fp32(a, b, n);
    const bool pass = std::isfinite(got) && std::fabs(static_cast<double>(got) - expected) <= tolerance;
    ++checks;
    if (!pass) {
        ++failures;
        std::printf("FAIL %-18s n=%llu got=%.9g expected=%.9g tolerance=%.3g\n", name,
                    static_cast<unsigned long long>(n), static_cast<double>(got), expected, tolerance);
    }
}

void generated_case(std::size_t n) {
    // Offset both pointers so the test does not accidentally require alignment.
    std::vector<float> a(n + 1), b(n + 1);
    double expected = 0.0;
    double sum_abs = 0.0;
    for (std::size_t x = 0; x < n; ++x) {
        a[x + 1] = static_cast<float>(static_cast<int>(x % 13) - 6) / 7.0f;
        b[x + 1] = static_cast<float>(static_cast<int>((3 * x + 2) % 17) - 8) / 9.0f;
        const double product = static_cast<double>(a[x + 1]) * static_cast<double>(b[x + 1]);
        expected += product;
        sum_abs += std::fabs(product);
    }

    // A conservative FP32 multiplication-and-summation bound, separate from
    // the proposal's fixed end-to-end output contract.
    const double eps = static_cast<double>(std::numeric_limits<float>::epsilon());
    const double neps = static_cast<double>(n) * eps;
    const double gamma = neps / (1.0 - neps);
    const double tolerance = 2.0 * gamma * sum_abs + 1e-7;
    check("unaligned_mixed", a.data() + 1, b.data() + 1, n, expected, tolerance);
}

void value_accumulation_case() {
    // Nine value dimensions exercise one AVX2 group plus a scalar tail.
    const float first[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float second[] = {2, 1, -1, 3, 2, 4, 1, 5, -2};
    const double expected[] = {1.7, 1.3, 0.2, 3.3, 2.9, 4.6, 2.8, 5.9, 1.3};
    std::vector<float> accum(10, 0.0f);
    attn::weighted_add_avx2_fp32(0.3f, first, accum.data() + 1, 9);
    attn::weighted_add_avx2_fp32(0.7f, second, accum.data() + 1, 9);
    for (std::size_t x = 0; x < 9; ++x) {
        ++checks;
        if (std::fabs(static_cast<double>(accum[x + 1]) - expected[x]) > 1e-6) {
            ++failures;
            std::printf("FAIL weighted_add lane=%llu got=%.9g expected=%.9g\n",
                        static_cast<unsigned long long>(x),
                        static_cast<double>(accum[x + 1]), expected[x]);
        }
    }
}

}  // namespace

int main() {
    const float a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const float b[] = {2, -1, 1, 2, 3, -2, 1, 2, -1, 1};
    check("worked_example", a, b, 10, 38.0, 0.0);

    const std::size_t lengths[] = {1, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33, 64, 65};
    for (std::size_t n : lengths) generated_case(n);

    value_accumulation_case();

    std::printf("%d checks, %d failures -> %s\n", checks, failures,
                failures == 0 ? "SIMD KERNEL SUITE PASSED" : "SIMD KERNEL SUITE FAILED");
    return failures == 0 ? 0 : 1;
}
