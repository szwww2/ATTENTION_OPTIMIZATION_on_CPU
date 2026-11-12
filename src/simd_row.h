#pragma once

#include "attn.h"

namespace attn {
namespace detail {

// Row-wise route: complete one query row before starting the next.
void row_attention_simd(const Shape& s, const float* q, const float* k, const float* v,
                        std::size_t i, float inv_sqrt_d, float* scores, float* out_row);

// Full-matrix route: each phase processes all query rows before the next phase.
void full_score_row_simd(const Shape& s, const float* q, const float* k,
                         std::size_t i, float inv_sqrt_d, float* scores);
void full_softmax_row_fp32(const Shape& s, float* scores);
void full_value_row_simd(const Shape& s, const float* v, const float* probs,
                         float* out_row);

}  // namespace detail
}  // namespace attn
