#pragma once

#include <cstddef>
#include <immintrin.h>

#if !defined(__AVX2__)
#error "attn_simd.h requires AVX2 (for example, compile with -mavx2)"
#endif

namespace attn {

// The pointers may be unaligned. Each helper includes its scalar tail.
// Inline definitions avoid an inner-loop function call for every score/value row.
inline float dot_avx2_fp32(const float* a, const float* b, std::size_t n) {
    __m256 partial = _mm256_setzero_ps();
    std::size_t x = 0;
    for (; n - x >= 8; x += 8) {
        const __m256 va = _mm256_loadu_ps(a + x);
        const __m256 vb = _mm256_loadu_ps(b + x);
        partial = _mm256_add_ps(partial, _mm256_mul_ps(va, vb));
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, partial);
    float sum = 0.0f;
    for (float lane : lanes) sum += lane;
    for (; x < n; ++x) sum += a[x] * b[x];
    return sum;
}

inline void weighted_add_avx2_fp32(float weight, const float* values,
                                   float* accum, std::size_t n) {
    const __m256 vw = _mm256_set1_ps(weight);
    std::size_t x = 0;
    for (; n - x >= 8; x += 8) {
        const __m256 vv = _mm256_loadu_ps(values + x);
        const __m256 va = _mm256_loadu_ps(accum + x);
        _mm256_storeu_ps(accum + x, _mm256_add_ps(va, _mm256_mul_ps(vw, vv)));
    }
    for (; x < n; ++x) accum[x] += weight * values[x];
}

}  // namespace attn
