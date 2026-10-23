// Scalar streaming: fixed FP64 contract, tile schedules and memory boundaries.
#include "attn.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
int checks = 0, failures = 0;
double max_error = 0.0;

// 汇总断言；记录失败但继续跑其余用例，无输入副作用。
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL: %s\n", message); }
}

// 给定shape与模式，生成一次输入/oracle；枚举块大小、顺序，检查精确scratch边界。
// kind=1等分数，2大分数差，3后续最大值上升的构造用例；只修改本地测试缓冲。
void exercise(attn::Shape s, unsigned seed, int kind = 0) {
    attn::Tensors t;
    t.resize(s);
    t.fill_qkv(seed);
    if (kind == 1) std::fill(t.q.begin(), t.q.end(), 0.0f);
    if (kind == 2) for (auto& x : t.q) x *= 100.0f;
    if (kind == 3) {
        // d=1；key分数逐渐上升，旧块有非零V，漏缩放旧u或l都会破坏结果。
        std::fill(t.q.begin(), t.q.end(), 1.0f);
        for (std::size_t j = 0; j < s.L; ++j) t.k[j] = float(j) * 2.0f - 8.0f;
        for (std::size_t j = 0; j < s.L; ++j)
            for (std::size_t y = 0; y < s.dv; ++y) t.v[j*s.dv+y] = float(j + y) - 3.0f;
    }
    const auto q = t.q, k = t.k, v = t.v;
    attn::oracle_fp64(s, q.data(), k.data(), v.data(), t.o_ref.data());
    if (kind == 1) {
        std::vector<double> analytic(t.o_ref.size());
        attn::uniform_attention_reference(s, v.data(), analytic.data());
        for (std::size_t i = 0; i < analytic.size(); ++i)
            check(std::abs(analytic[i] - t.o_ref[i]) < 1e-12, "uniform oracle");
    }
    const std::size_t qs[] = {0, 1, 3, 16, s.L + 3};
    const std::size_t ks[] = {0, 1, 2, 7, 128, 256, 512, s.L + 5};
    for (auto bq : qs) for (auto bk : ks) {
        const auto eq = std::min(s.L, std::max(std::size_t{1}, bq));
        const auto ek = std::min(s.L, std::max(std::size_t{1}, bk));
        const auto size = attn::blocked_streaming_workspace_elems(s, bq, bk);
        check(size == eq*ek + eq*s.dv + 2*eq, "workspace formula");
        // 两端哨兵＋NaN内部，检测越界与未初始化读取；只传有效区域给kernel。
        std::vector<float> scratch(size + 2, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> output(t.o.size() + 2, std::numeric_limits<float>::quiet_NaN());
        scratch.front() = scratch.back() = 76543.0f;
        output.front() = output.back() = 98765.0f;
        attn::blocked_streaming_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(),
                                           bq, bk, scratch.data()+1, output.data()+1);
        auto result = attn::check_contract(output.data()+1, t.o_ref.data(), t.o.size(), {});
        check(result.pass, "natural order vs FP64");
        max_error = std::max(max_error, result.max_abs);
        const std::vector<float> first(output.begin()+1, output.end()-1);
        std::vector<std::size_t> order(s.L/ek + (s.L%ek != 0));
        std::iota(order.begin(), order.end(), 0);
        for (int schedule = 0; schedule < 3; ++schedule) {
            if (schedule == 1) std::reverse(order.begin(), order.end());
            if (schedule == 2) { std::mt19937 rng(seed); std::shuffle(order.begin(), order.end(), rng); }
            std::fill(scratch.begin()+1, scratch.end()-1, std::numeric_limits<float>::quiet_NaN());
            std::fill(output.begin()+1, output.end()-1, std::numeric_limits<float>::quiet_NaN());
            attn::blocked_streaming_scalar_ordered_fp32(s, t.q.data(), t.k.data(), t.v.data(),
                bq, bk, order.data(), order.size(), scratch.data()+1, output.data()+1);
            result = attn::check_contract(output.data()+1, t.o_ref.data(), t.o.size(), {});
            check(result.pass, "key permutation vs FP64");
            max_error = std::max(max_error, result.max_abs);
            if (schedule == 0) check(std::equal(first.begin(), first.end(), output.begin()+1),
                                     "repeated natural call overwrites state identically");
            check(scratch.front()==76543.0f && scratch.back()==76543.0f, "scratch guards");
            check(output.front()==98765.0f && output.back()==98765.0f, "output guards");
        }
        check(t.q==q && t.k==k && t.v==v, "inputs unchanged");
    }
}

// 检查非法shape、溢出、错误排列/空指针在输出写入前被拒绝，不掩盖其他异常。
void invalid_arguments() {
    for (attn::Shape s : {attn::Shape{0,1,1}, {1,0,1}, {1,1,0}}) {
        bool rejected=false;
        try { attn::blocked_streaming_workspace_elems(s, 1, 1); }
        catch (const std::invalid_argument&) { rejected=true; }
        check(rejected, "zero dimension rejected");
    }
    bool overflow=false;
    try { attn::blocked_streaming_workspace_elems({std::numeric_limits<std::size_t>::max(),2,1},1,1); }
    catch (const std::overflow_error&) { overflow=true; }
    check(overflow, "overflow rejected");
    attn::Shape s{5,1,1};
    float q[5]={}, k[5]={}, v[5]={}, out[5]={9,9,9,9,9};
    std::vector<float> ws(attn::blocked_streaming_workspace_elems(s,2,2), 42.0f);
    const std::vector<std::vector<std::size_t>> bad={{0,0,2},{0,1,3},{0,1},{0,1,2,3}};
    for (const auto& order : bad) {
        bool rejected=false;
        try { attn::blocked_streaming_scalar_ordered_fp32(s,q,k,v,2,2,order.data(),order.size(),ws.data(),out); }
        catch (const std::invalid_argument&) { rejected=true; }
        check(rejected, "invalid permutation rejected");
        check(std::all_of(out,out+5,[](float x){return x==9;}), "invalid schedule leaves output intact");
        check(std::all_of(ws.begin(),ws.end(),[](float x){return x==42;}), "invalid schedule leaves scratch intact");
    }
    bool null_rejected=false;
    try { attn::blocked_streaming_scalar_fp32(s,nullptr,k,v,2,2,ws.data(),out); }
    catch (const std::invalid_argument&) { null_rejected=true; }
    check(null_rejected, "null buffer rejected");
}
}  // namespace

// 执行不同shape/seed与数值构造；失败返回非零，供脚本和CI使用。
int main() {
    for (unsigned seed : {1u,2u,3u})
        for (attn::Shape s : {attn::Shape{1,64,64},{7,31,33},{127,64,64},
                              {128,64,64},{129,65,64},{256,64,33}})
            exercise(s,seed);
    exercise({17,31,33},7,1);
    exercise({33,64,33},7,2);
    exercise({9,1,3},7,3);
    invalid_arguments();
    std::printf("streaming: %d checks, %d failures, max_abs=%.9g -> %s\n",
                checks, failures, max_error, failures ? "FAILED" : "STREAMING SUITE PASSED");
    return failures ? 1 : 0;
}
