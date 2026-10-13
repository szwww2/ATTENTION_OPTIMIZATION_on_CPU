// Shared interface for the attention routes, for the other teammates to implement.
// O = softmax(QK^T/sqrt(d))V, FP32, non-causal, single head, batch 1.
// Checked against the FP64 oracle: |O - O64| <= 1e-4 + 1e-3|O64|.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace attn {

struct Shape {
    std::size_t L = 0;   // sequence length
    std::size_t d = 0;   // query/key dimension
    std::size_t dv = 0;  // value dimension
};

inline std::size_t qk_elems(const Shape& s) { return s.L * s.d; }
inline std::size_t v_elems(const Shape& s) { return s.L * s.dv; }
// Workspace size in floats.
inline std::size_t full_matrix_workspace_elems(const Shape& s) { return s.L * s.L; }

// Buffers for one call.
struct Tensors {
    std::vector<float> q, k, v;   // inputs
    std::vector<float> ws;        // workspace
    std::vector<float> o;         // FP32 output
    std::vector<double> o_ref;    // FP64 reference output for validation

    void resize(const Shape& s);
    void fill_device_neutral(std::uint64_t seed);
};

// Own RNG, splitmix64: the same seed must give the same inputs on any machine
// (std::uniform_real_distribution does not guarantee that).
class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next_u64();

private:
    std::uint64_t state_;
};

void fill_uniform(float* dst, std::size_t n, std::uint64_t seed, float lo, float hi);

// Route 0: FP64 reference. Never timed.
void oracle_fp64(const Shape& s, const float* q, const float* k, const float* v, double* out);

// Route 1: full matrix. S = QK^T/sqrt(d) stored whole, then softmax rows, then PV.
void full_matrix_scalar_fp32(const Shape& s, const float* q, const float* k, const float* v,
                             float* score_ws, float* out);

// q = 0: every output row is the mean of the V rows. Used by the equal-scores case.
void uniform_attention_reference(const Shape& s, const float* v, double* out);

// Pass rule. max_rel blows up near zero, so report it next to max_abs.
struct Contract {
    double atol = 1e-4;
    double rtol = 1e-3;
    double rel_denom_floor = 1e-8;  // relative-error denominator: max(1e-8, |ref|)
};

struct ErrorReport {
    double max_abs = 0.0;
    double max_rel = 0.0;
    // <= 0 means pass
    double worst_margin = 0.0;
    std::size_t worst_index = 0;
    std::size_t non_finite = 0;
    bool pass = false;
};

ErrorReport check_contract(const float* got, const double* ref, std::size_t n, const Contract& c);

}  // namespace attn
