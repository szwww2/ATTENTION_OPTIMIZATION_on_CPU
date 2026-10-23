// Shared interface for the attention routes.
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
inline std::size_t rowwise_workspace_elems(const Shape& s, std::size_t query_block) {
    return query_block * s.L;
}

// Buffers for one call.
struct Tensors {
    std::vector<float> q, k, v;   // inputs
    std::vector<float> ws;        // workspace, sized by the caller for the route it runs
    std::vector<float> o;         // FP32 output
    std::vector<double> o_ref;    // FP64 reference output for validation

    void resize(const Shape& s);
    void fill_qkv(std::uint64_t seed);
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

// Route 2: row-wise. Only a Bq x L window is kept; same result as route 1.
void rowwise_scalar_fp32(const Shape& s, const float* q, const float* k, const float* v,
                         std::size_t query_block, float* score_ws, float* out);

// Route 3: scalar blocked streaming. Scratch counts ALL state, in float elements.
// 正尺寸；Bq/Bk=0按1处理，超过L截为L。非法尺寸/溢出抛出异常。
std::size_t blocked_streaming_workspace_elems(const Shape& s,
                                              std::size_t query_block,
                                              std::size_t key_block);

// 输入只读、各缓冲区不得重叠；调用者按helper预分配scratch和L*dv输出。
// 有效调用不分配内存，无I/O；覆盖out，scratch初值任意，所有状态在调用内初始化。
void blocked_streaming_scalar_fp32(const Shape& s,
                                   const float* q, const float* k, const float* v,
                                   std::size_t query_block, std::size_t key_block,
                                   float* workspace, float* out);

// 测试专用：key_blocks为[0,ceil(L/effective_Bk))的完整排列。
// 在写输出前验证数量/范围/重复；不合法则抛invalid_argument。
// 验证使用O(n_blocks^2)比较、不分配内存；性能测试使用上面的自然顺序入口。
void blocked_streaming_scalar_ordered_fp32(const Shape& s,
                                           const float* q, const float* k, const float* v,
                                           std::size_t query_block, std::size_t key_block,
                                           const std::size_t* key_blocks,
                                           std::size_t n_blocks,
                                           float* workspace, float* out);

// Route 3优化入口：helper复用单线程全部scratch，AVX2只在独立编译单元开启。
// CPU/OS不支持AVX2时返回false；计算入口会在写缓冲区前抛runtime_error。
bool blocked_streaming_avx2_supported();
void blocked_streaming_simd_fp32(const Shape& s,
                                 const float* q, const float* k, const float* v,
                                 std::size_t query_block, std::size_t key_block,
                                 float* workspace, float* out);

// threads必须>0；返回threads份独立scratch的总float数，检查总字节溢出。
std::size_t blocked_streaming_omp_workspace_elems(const Shape& s,
                                                 std::size_t query_block,
                                                 std::size_t key_block, int threads);
// 静态分配query块，线程私有scratch；每个query在一个线程上完成所有key块。
// actual_threads可为空，否则写入本次实际team大小（可能小于请求值）。
// 不改变全局OpenMP设置；所有缓冲区/actual_threads必须互不重叠。
// 应用代码不分配，OpenMP运行时可自行分配；计时须包含parallel region。
void blocked_streaming_simd_omp_fp32(const Shape& s,
                                     const float* q, const float* k, const float* v,
                                     std::size_t query_block, std::size_t key_block,
                                     int threads, float* workspace, float* out,
                                     int* actual_threads = nullptr);

// Test only: the row order comes from the caller.
void rowwise_scalar_rows_fp32(const Shape& s, const float* q, const float* k, const float* v,
                              const std::size_t* rows, std::size_t n_rows, float* score_ws,
                              float* out);

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
    std::size_t non_finite = 0;
    bool pass = false;
};

ErrorReport check_contract(const float* got, const double* ref, std::size_t n, const Contract& c);

}  // namespace attn
