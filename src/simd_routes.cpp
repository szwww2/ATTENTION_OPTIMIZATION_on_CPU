// Full-matrix and row-wise attention using the same explicit AVX2 inner loops.
#include "attn.h"
#include "attn_simd.h"
#include "simd_row.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace attn {
namespace detail {

void row_attention_simd(const Shape& s, const float* q, const float* k, const float* v,
                        std::size_t i, float inv_sqrt_d, float* scores, float* out_row) {
    float row_max = -std::numeric_limits<float>::infinity();
    const float* q_row = q + i * s.d;
    for (std::size_t j = 0; j < s.L; ++j) {
        const float score = dot_avx2_fp32(q_row, k + j * s.d, s.d) * inv_sqrt_d;
        scores[j] = score;
        if (score > row_max) row_max = score;
    }

    float denom = 0.0f;
    for (std::size_t j = 0; j < s.L; ++j) {
        const float weight = std::exp(scores[j] - row_max);
        scores[j] = weight;
        denom += weight;
    }

    std::fill(out_row, out_row + s.dv, 0.0f);
    for (std::size_t j = 0; j < s.L; ++j) {
        weighted_add_avx2_fp32(scores[j], v + j * s.dv, out_row, s.dv);
    }
    for (std::size_t y = 0; y < s.dv; ++y) out_row[y] /= denom;
}

void full_score_row_simd(const Shape& s, const float* q, const float* k,
                         std::size_t i, float inv_sqrt_d, float* scores) {
    const float* q_row = q + i * s.d;
    for (std::size_t j = 0; j < s.L; ++j)
        scores[j] = dot_avx2_fp32(q_row, k + j * s.d, s.d) * inv_sqrt_d;
}

void full_softmax_row_fp32(const Shape& s, float* scores) {
    float row_max = -std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j < s.L; ++j)
        if (scores[j] > row_max) row_max = scores[j];

    float denom = 0.0f;
    for (std::size_t j = 0; j < s.L; ++j) {
        scores[j] = std::exp(scores[j] - row_max);
        denom += scores[j];
    }
    for (std::size_t j = 0; j < s.L; ++j) scores[j] /= denom;
}

void full_value_row_simd(const Shape& s, const float* v, const float* probs,
                         float* out_row) {
    std::fill(out_row, out_row + s.dv, 0.0f);
    for (std::size_t j = 0; j < s.L; ++j)
        weighted_add_avx2_fp32(probs[j], v + j * s.dv, out_row, s.dv);
}

}  // namespace detail

void full_matrix_simd_fp32(const Shape& s, const float* q, const float* k, const float* v,
                           float* score_ws, float* out) {
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(s.d));
    for (std::size_t i = 0; i < s.L; ++i)
        detail::full_score_row_simd(s, q, k, i, inv_sqrt_d, score_ws + i * s.L);
    for (std::size_t i = 0; i < s.L; ++i)
        detail::full_softmax_row_fp32(s, score_ws + i * s.L);
    for (std::size_t i = 0; i < s.L; ++i)
        detail::full_value_row_simd(s, v, score_ws + i * s.L, out + i * s.dv);
}

void rowwise_simd_fp32(const Shape& s, const float* q, const float* k, const float* v,
                       std::size_t query_block, float* score_ws, float* out) {
    const std::size_t bq = (query_block == 0) ? 1 : query_block;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(s.d));
    for (std::size_t i0 = 0; i0 < s.L; i0 += bq) {
        const std::size_t i1 = std::min(i0 + bq, s.L);
        for (std::size_t i = i0; i < i1; ++i) {
            detail::row_attention_simd(s, q, k, v, i, inv_sqrt_d,
                                       score_ws + (i - i0) * s.L, out + i * s.dv);
        }
    }
}

}  // namespace attn
