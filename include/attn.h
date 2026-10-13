// Shared interface for the attention routes, for the other teammates to implement.
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
inline std::size_t out_elems(const Shape& s) { return s.L * s.dv; }
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

}  // namespace attn
