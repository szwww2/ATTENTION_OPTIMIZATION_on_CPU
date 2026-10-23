// Route 3: query/key tiling with stable online softmax; scalar FP32 only.
#include "attn.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace attn {
namespace {

// 检查元素数乘法；返回a*b，不访问数据，溢出时拒绝错误的缓冲区大小。
std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("streaming element count overflow");
    return a * b;
}

// 检查各scratch分区之和；不访问数据，溢出抛异常。
std::size_t checked_add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("streaming workspace overflow");
    return a + b;
}

struct Layout {
    std::size_t bq, bk, scores, states, total;
};

// 正尺寸前提检查并计算布局：score[bq*bk],m[bq],l[bq],u[bq*dv]。
// 将0块参数规范化为1，超大块截为L；返回精确元素数，无分配/副作用。
Layout layout(const Shape& s, std::size_t query_block, std::size_t key_block) {
    if (!s.L || !s.d || !s.dv)
        throw std::invalid_argument("streaming requires positive L, d and dv");
    checked_mul(checked_mul(s.L, s.d), sizeof(float));
    checked_mul(checked_mul(s.L, s.dv), sizeof(float));
    const std::size_t bq = std::min(s.L, std::max(std::size_t{1}, query_block));
    const std::size_t bk = std::min(s.L, std::max(std::size_t{1}, key_block));
    const std::size_t scores = checked_mul(bq, bk);
    const std::size_t states = checked_mul(bq, s.dv);
    const std::size_t total = checked_add(checked_add(scores, checked_mul(2, bq)), states);
    checked_mul(total, sizeof(float));
    return {bq, bk, scores, states, total};
}

// 核心计算：已验证的布局与可选key块排列；写workspace/out，输入只读。
// order=nullptr代表自然顺序。每个key块在进入下一个块前服务所有当前query。
void run(const Shape& s, const Layout& w, const float* q, const float* k,
         const float* v, const std::size_t* order, float* workspace, float* out) {
    if (!q || !k || !v || !workspace || !out)
        throw std::invalid_argument("streaming received a null data buffer");
    float* score = workspace;
    float* maximum = score + w.scores;
    float* normalizer = maximum + w.bq;
    float* accum = normalizer + w.bq;
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.d));
    const std::size_t blocks = s.L / w.bk + (s.L % w.bk != 0);

    for (std::size_t i0 = 0; i0 < s.L;) {
        const std::size_t nq = std::min(w.bq, s.L - i0);
        // 每个query块独立开始；不能依赖调用者清零，重复调用也必须覆盖状态。
        std::fill(accum, accum + nq * s.dv, 0.0f);
        std::fill(normalizer, normalizer + nq, 0.0f);

        for (std::size_t step = 0; step < blocks; ++step) {
            const std::size_t block = order ? order[step] : step;
            const std::size_t j0 = block * w.bk;
            const std::size_t nk = std::min(w.bk, s.L - j0);

            // key在外、query在内：同一K行连续供nq个query打分，score只留当前tile。
            for (std::size_t j = 0; j < nk; ++j) {
                const float* key = k + (j0 + j) * s.d;
                for (std::size_t r = 0; r < nq; ++r) {
                    const float* query = q + (i0 + r) * s.d;
                    float dot = 0.0f;
                    for (std::size_t x = 0; x < s.d; ++x) dot += query[x] * key[x];
                    score[r * w.bk + j] = dot * scale;
                }
            }

            for (std::size_t r = 0; r < nq; ++r) {
                float* row = score + r * w.bk;
                float block_max = row[0];  // nk>=1，尾块也只读取有效分数。
                for (std::size_t j = 1; j < nk; ++j) block_max = std::max(block_max, row[j]);
                float new_max = block_max;
                if (step != 0) {
                    new_max = std::max(maximum[r], block_max);
                    const float alpha = std::exp(maximum[r] - new_max);
                    normalizer[r] *= alpha;
                    // m改变时，旧分母和旧加权和必须同步缩放，否则输出错误。
                    for (std::size_t y = 0; y < s.dv; ++y) accum[r * s.dv + y] *= alpha;
                }
                // 首块直接用其最大值初始化，不计算-inf-(-inf)。
                maximum[r] = new_max;
                float block_sum = 0.0f;
                for (std::size_t j = 0; j < nk; ++j) {
                    row[j] = std::exp(row[j] - new_max);
                    block_sum += row[j];
                }
                normalizer[r] += block_sum;
            }

            // 当前V行供整个query块复用；未归一化的权重只消费一次，不写全局P。
            for (std::size_t j = 0; j < nk; ++j) {
                const float* value = v + (j0 + j) * s.dv;
                for (std::size_t r = 0; r < nq; ++r) {
                    const float weight = score[r * w.bk + j];
                    float* dst = accum + r * s.dv;
                    for (std::size_t y = 0; y < s.dv; ++y) dst[y] += weight * value[y];
                }
            }
        }
        // 扫完所有key块才归一化；每行完整覆盖输出，不累加到上次结果。
        for (std::size_t r = 0; r < nq; ++r)
            for (std::size_t y = 0; y < s.dv; ++y)
                out[(i0 + r) * s.dv + y] = accum[r * s.dv + y] / normalizer[r];
        i0 += nq;
    }
}
}  // namespace

// 返回单次标量调用所需的全部scratch元素数；无数据访问或动态分配。
std::size_t blocked_streaming_workspace_elems(const Shape& s, std::size_t bq, std::size_t bk) {
    return layout(s, bq, bk).total;
}

// 自然key块顺序的生产入口；校验尺寸/空指针后写输出，合法路径不分配。
void blocked_streaming_scalar_fp32(const Shape& s, const float* q, const float* k,
                                   const float* v, std::size_t bq, std::size_t bk,
                                   float* workspace, float* out) {
    run(s, layout(s, bq, bk), q, k, v, nullptr, workspace, out);
}

// 测试入口：排列必须恰好覆盖所有key块。先完整验证，再允许写scratch/output。
void blocked_streaming_scalar_ordered_fp32(const Shape& s, const float* q, const float* k,
                                           const float* v, std::size_t bq, std::size_t bk,
                                           const std::size_t* order, std::size_t count,
                                           float* workspace, float* out) {
    const Layout w = layout(s, bq, bk);
    const std::size_t expected = s.L / w.bk + (s.L % w.bk != 0);
    if (!order || count != expected)
        throw std::invalid_argument("streaming key block count mismatch");
    for (std::size_t i = 0; i < count; ++i) {
        if (order[i] >= expected) throw std::invalid_argument("streaming key block out of range");
        for (std::size_t j = 0; j < i; ++j)
            if (order[i] == order[j]) throw std::invalid_argument("duplicate streaming key block");
    }
    run(s, w, q, k, v, order, workspace, out);
}
}  // namespace attn
