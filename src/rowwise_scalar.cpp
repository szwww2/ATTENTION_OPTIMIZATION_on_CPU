// Route 2: row-wise scalar. A row is computed exactly like in route 1, so results match.
#include "attn.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace attn {
namespace {

// One row: scores, softmax, then P*V.
void row_attention(const Shape& s, const float* q, const float* k, const float* v, std::size_t i,
                   float inv_sqrt_d, float* score, float* out) {
    const std::size_t L = s.L;
    const std::size_t d = s.d;
    const std::size_t dv = s.dv;

    for (std::size_t j = 0; j < L; ++j) {
        float acc = 0.0f;
        for (std::size_t x = 0; x < d; ++x) {
            acc += q[i * d + x] * k[j * d + x];
        }
        score[j] = acc * inv_sqrt_d;
    }

    float row_max = -std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j < L; ++j)
        if (score[j] > row_max) row_max = score[j];
    float denom = 0.0f;
    for (std::size_t j = 0; j < L; ++j) {
        const float pj = std::exp(score[j] - row_max);
        score[j] = pj;
        denom += pj;
    }
    for (std::size_t j = 0; j < L; ++j) score[j] /= denom;

    for (std::size_t y = 0; y < dv; ++y) {
        float acc = 0.0f;
        for (std::size_t j = 0; j < L; ++j) {
            acc += score[j] * v[j * dv + y];
        }
        out[i * dv + y] = acc;
    }
}

}  // namespace

void rowwise_scalar_fp32(const Shape& s, const float* q, const float* k, const float* v,
                         std::size_t query_block, float* score_ws, float* out) {
    const std::size_t L = s.L;
    const std::size_t bq = (query_block == 0) ? 1 : query_block;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(s.d));

    for (std::size_t i0 = 0; i0 < L; i0 += bq) {
        const std::size_t i1 = std::min(i0 + bq, L);  // the last block may be a partial one
        for (std::size_t i = i0; i < i1; ++i) {
            row_attention(s, q, k, v, i, inv_sqrt_d, score_ws + (i - i0) * L, out);
        }
    }
}

void rowwise_scalar_rows_fp32(const Shape& s, const float* q, const float* k, const float* v,
                              const std::size_t* rows, std::size_t n_rows, float* score_ws,
                              float* out) {
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(s.d));
    for (std::size_t r = 0; r < n_rows; ++r) {
        row_attention(s, q, k, v, rows[r], inv_sqrt_d, score_ws, out);
    }
}

}  // namespace attn
