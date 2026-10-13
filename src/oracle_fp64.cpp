// FP64 reference: no blocking, no SIMD, no code shared with the FP32 kernels.
#include "attn.h"

#include <cmath>
#include <limits>
#include <vector>

namespace attn {

void oracle_fp64(const Shape& s, const float* q, const float* k, const float* v, double* out) {
    const std::size_t L = s.L;
    const std::size_t d = s.d;
    const std::size_t dv = s.dv;
    const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(d));

    std::vector<double> score(L);
    for (std::size_t i = 0; i < L; ++i) {
        double row_max = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < L; ++j) {
            double acc = 0.0;
            for (std::size_t x = 0; x < d; ++x) {
                acc += static_cast<double>(q[i * d + x]) * static_cast<double>(k[j * d + x]);
            }
            score[j] = acc * inv_sqrt_d;
            if (score[j] > row_max) row_max = score[j];
        }

        double denom = 0.0;
        for (std::size_t j = 0; j < L; ++j) {
            score[j] = std::exp(score[j] - row_max);
            denom += score[j];
        }

        for (std::size_t y = 0; y < dv; ++y) {
            double acc = 0.0;
            for (std::size_t j = 0; j < L; ++j) {
                acc += score[j] * static_cast<double>(v[j * dv + y]);
            }
            out[i * dv + y] = acc / denom;
        }
    }
}

void uniform_attention_reference(const Shape& s, const float* v, double* out) {
    // q = 0: each row is the mean of V
    for (std::size_t y = 0; y < s.dv; ++y) {
        double acc = 0.0;
        for (std::size_t j = 0; j < s.L; ++j) acc += static_cast<double>(v[j * s.dv + y]);
        const double mean = acc / static_cast<double>(s.L);
        for (std::size_t i = 0; i < s.L; ++i) out[i * s.dv + y] = mean;
    }
}

}  // namespace attn
