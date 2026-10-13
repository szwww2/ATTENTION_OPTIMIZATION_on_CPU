// Deterministic inputs: splitmix64, uniform in [-1, 1].
#include "attn.h"

namespace attn {

void Tensors::resize(const Shape& s) {
    q.assign(qk_elems(s), 0.0f);
    k.assign(qk_elems(s), 0.0f);
    v.assign(v_elems(s), 0.0f);
    ws.assign(full_matrix_workspace_elems(s), 0.0f);
    o.assign(out_elems(s), 0.0f);
    o_ref.assign(out_elems(s), 0.0);
}

std::uint64_t SplitMix64::next_u64() {
    state_ += 0x9E3779B97F4A7C15ULL;  // odd step, so the state covers all 2^64 values
    std::uint64_t z = state_;
    // two rounds of xor-shift + multiply: hides the fixed step; the constants are fixed by
    // the algorithm (splitmix64), so do not tune them
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double SplitMix64::next_uniform(double lo, double hi) {
    // top 53 bits -> [0, 1)
    const double u = static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    return lo + (hi - lo) * u;
}

void fill_uniform(float* dst, std::size_t n, std::uint64_t seed, float lo, float hi) {
    SplitMix64 rng(seed);
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(rng.next_uniform(static_cast<double>(lo), static_cast<double>(hi)));
    }
}

// Distinct streams per tensor, so one seed determines the whole call.
void Tensors::fill_device_neutral(std::uint64_t seed) {
    fill_uniform(q.data(), q.size(), seed * 3u + 1u, -1.0f, 1.0f);
    fill_uniform(k.data(), k.size(), seed * 3u + 2u, -1.0f, 1.0f);
    fill_uniform(v.data(), v.size(), seed * 3u + 3u, -1.0f, 1.0f);
}

}  // namespace attn
