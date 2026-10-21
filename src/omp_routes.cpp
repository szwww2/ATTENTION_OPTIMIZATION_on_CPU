// Static OpenMP scheduling across independent query blocks.
#include "attn.h"
#include "simd_row.h"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace attn {

void full_matrix_simd_omp_fp32(const Shape& s, const float* q, const float* k, const float* v,
                               std::size_t query_block, int threads,
                               float* score_ws, float* out) {
    const std::size_t bq = (query_block == 0) ? 1 : query_block;
    const int workers = (threads > 0) ? threads : 1;
    const int blocks = static_cast<int>(s.L / bq + (s.L % bq != 0));
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(s.d));

#pragma omp parallel num_threads(workers)
    {
        // Each omp-for has an implicit barrier: all scores, then all P, then O.
#pragma omp for schedule(static)
        for (int block = 0; block < blocks; ++block) {
            const std::size_t i0 = static_cast<std::size_t>(block) * bq;
            const std::size_t i1 = std::min(i0 + bq, s.L);
            for (std::size_t i = i0; i < i1; ++i)
                detail::full_score_row_simd(s, q, k, i, inv_sqrt_d, score_ws + i * s.L);
        }
#pragma omp for schedule(static)
        for (int block = 0; block < blocks; ++block) {
            const std::size_t i0 = static_cast<std::size_t>(block) * bq;
            const std::size_t i1 = std::min(i0 + bq, s.L);
            for (std::size_t i = i0; i < i1; ++i)
                detail::full_softmax_row_fp32(s, score_ws + i * s.L);
        }
#pragma omp for schedule(static)
        for (int block = 0; block < blocks; ++block) {
            const std::size_t i0 = static_cast<std::size_t>(block) * bq;
            const std::size_t i1 = std::min(i0 + bq, s.L);
            for (std::size_t i = i0; i < i1; ++i)
                detail::full_value_row_simd(s, v, score_ws + i * s.L, out + i * s.dv);
        }
    }
}

void rowwise_simd_omp_fp32(const Shape& s, const float* q, const float* k, const float* v,
                           std::size_t query_block, int threads,
                           float* score_ws, float* out) {
    const std::size_t bq = (query_block == 0) ? 1 : query_block;
    const int workers = (threads > 0) ? threads : 1;
    const int blocks = static_cast<int>(s.L / bq + (s.L % bq != 0));
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(s.d));

#pragma omp parallel num_threads(workers)
    {
        const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
        float* local_scores = score_ws + tid * bq * s.L;
#pragma omp for schedule(static)
        for (int block = 0; block < blocks; ++block) {
            const std::size_t i0 = static_cast<std::size_t>(block) * bq;
            const std::size_t i1 = std::min(i0 + bq, s.L);
            for (std::size_t i = i0; i < i1; ++i) {
                detail::row_attention_simd(s, q, k, v, i, inv_sqrt_d,
                                           local_scores + (i - i0) * s.L,
                                           out + i * s.dv);
            }
        }
    }
}

}  // namespace attn
