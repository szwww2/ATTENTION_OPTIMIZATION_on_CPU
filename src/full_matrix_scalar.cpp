// Route 1: full-matrix scalar. Baseline, not the oracle.
#include "attn.h"

#include <cmath>
#include <limits>

namespace attn {

void full_matrix_scalar_fp32(const Shape& s, const float* q, const float* k, const float* v,
                             float* score_ws, float* out) {
    const std::size_t L = s.L;
    const std::size_t d = s.d;
    const std::size_t dv = s.dv;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));

    // Stage 1: materialize every score before applying any softmax or P*V.
    for (std::size_t i = 0; i < L; ++i) {
        float* score = score_ws + i * L;
        for (std::size_t j = 0; j < L; ++j) {
            float acc = 0.0f;
            for (std::size_t x = 0; x < d; ++x) {
                acc += q[i * d + x] * k[j * d + x];
            }
            score[j] = acc * inv_sqrt_d;
        }
    }

    // Stage 2: overwrite the complete score matrix with row-normalized P.
    for (std::size_t i = 0; i < L; ++i) {
        float* score = score_ws + i * L;
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
    }

    // Stage 3: read the materialized P matrix to compute all output rows.
    for (std::size_t i = 0; i < L; ++i) {
        const float* score = score_ws + i * L;
        for (std::size_t y = 0; y < dv; ++y) {
            float acc = 0.0f;
            for (std::size_t j = 0; j < L; ++j) {
                acc += score[j] * v[j * dv + y];
            }
            out[i * dv + y] = acc;
        }
    }
}

}  // namespace attn
