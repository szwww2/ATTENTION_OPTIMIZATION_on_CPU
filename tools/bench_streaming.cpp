// Route-3-only pilot. Natural-order scalar kernel; no SIMD or thread integration.
#include "attn.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Sample { std::size_t bk, bytes; int round, order; double ms; };

// 一个shape共用输入和oracle；各候选预分配完整scratch，计时前后校验。
// 每轮轮换Bk顺序以减少固定执行顺序偏差；输出CSV仅在本shape所有校验通过后发生。
void measure(attn::Shape s, const std::vector<std::size_t>& candidates) {
    constexpr int warmup=10, reps=20;
    constexpr std::uint64_t seed=20261022;
    constexpr std::size_t bq=16;
    attn::Tensors t;
    t.resize(s);
    t.fill_qkv(seed);
    const auto q=t.q, k=t.k, v=t.v;
    attn::oracle_fp64(s,t.q.data(),t.k.data(),t.v.data(),t.o_ref.data());
    std::vector<std::vector<float>> scratch, output;
    std::vector<attn::ErrorReport> before, after;
    std::vector<Sample> samples;
    samples.reserve(reps*candidates.size());
    for (auto bk : candidates) {
        scratch.emplace_back(attn::blocked_streaming_workspace_elems(s,bq,bk));
        output.emplace_back(t.o.size());
    }
    // 调用时初始化状态属于kernel开销；输入、workspace/output分配及oracle不计时。
    const auto call = [&](std::size_t n) {
        attn::blocked_streaming_scalar_fp32(s,t.q.data(),t.k.data(),t.v.data(),bq,
            candidates[n],scratch[n].data(),output[n].data());
    };
    const auto validate = [&](std::size_t n) {
        const auto e=attn::check_contract(output[n].data(),t.o_ref.data(),t.o.size(),{});
        if (!e.pass) throw std::runtime_error("streaming FP64 contract failed");
        return e;
    };
    for (std::size_t n=0;n<candidates.size();++n) { call(n); before.push_back(validate(n)); }
    for (int round=0;round<warmup;++round)
        for (std::size_t n=0;n<candidates.size();++n) call((n+round)%candidates.size());
    for (int round=0;round<reps;++round) {
        for (std::size_t order=0;order<candidates.size();++order) {
            const auto n=(order+round)%candidates.size();
            const auto start=std::chrono::steady_clock::now();
            call(n);
            const auto stop=std::chrono::steady_clock::now();
            const double ms=std::chrono::duration<double,std::milli>(stop-start).count();
            if (!std::isfinite(ms) || ms<=0) throw std::runtime_error("invalid timer sample");
            samples.push_back({candidates[n],scratch[n].size()*sizeof(float),round,int(order),ms});
        }
    }
    for (std::size_t n=0;n<candidates.size();++n) after.push_back(validate(n));
    if (q!=t.q || k!=t.k || v!=t.v) throw std::runtime_error("input mutation");
    for (const auto& r : samples) {
        const auto n=std::find(candidates.begin(),candidates.end(),r.bk)-candidates.begin();
        std::printf("blocked_streaming,scalar,1,%zu,%zu,%zu,%llu,%zu,%zu,%zu,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,1\n",
            s.L,s.d,s.dv,static_cast<unsigned long long>(seed),bq,r.bk,r.bytes,
            warmup,reps,r.round,r.order,r.ms,before[n].max_abs,before[n].max_rel,
            after[n].max_abs,after[n].max_rel);
    }
}
}  // namespace

// tune比较三个预先声明的Bk；grid使用选定Bk跑六形状；错误返回非零由上层拒绝数据。
int main(int argc,char** argv) {
    try {
        if (argc<2) throw std::invalid_argument("usage: attn_streaming_bench tune | grid <128|256|512>");
        const std::string mode=argv[1];
        std::size_t bk=0;
        if (mode=="grid" && argc==3) {
            const std::string arg=argv[2];
            if (arg!="128" && arg!="256" && arg!="512") throw std::invalid_argument("invalid Bk");
            bk=std::stoul(arg);
        } else if (mode!="tune" || argc!=2) throw std::invalid_argument("invalid arguments");
        std::puts("route,mode,threads,L,d,dv,seed,query_block,key_block,workspace_bytes,warmup,reps,sample,order,ms,pre_max_abs,pre_max_rel,post_max_abs,post_max_rel,contract_pass");
        if (mode=="tune") measure({2048,64,64},{128,256,512});
        else for (attn::Shape s : {attn::Shape{128,64,64},{512,64,64},{2048,64,64},
                                   {4096,64,64},{512,32,32},{2048,32,32}}) measure(s,{bk});
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"streaming benchmark: %s\n",e.what());
        return 1;
    }
}
