#include "attn.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
struct Config { const char* mode; int threads; };
struct Sample { std::size_t config; int round,order,actual; double ms; };
// 同shape共用输入/oracle；配置轮换顺序，预分配和校验均在计时外。
void measure(attn::Shape s) {
    const std::size_t bq=16,bk=256;
    constexpr int warmup=10,reps=20;
    std::vector<Config> configs={{"scalar",1},{"simd",1},{"simd_omp",1}};
    if (s.L==128 || s.L==4096)
        for (int t:{2,4,8}) configs.push_back({"simd_omp",t});
    attn::Tensors t; t.resize(s); t.fill_qkv(20261022);
    const auto q=t.q,k=t.k,v=t.v;
    attn::oracle_fp64(s,q.data(),k.data(),v.data(),t.o_ref.data());
    std::vector<std::vector<float>> ws,out;
    std::vector<attn::ErrorReport> before,after;
    std::vector<Sample> samples; samples.reserve(reps*configs.size());
    for (auto c:configs) {
        ws.emplace_back(attn::blocked_streaming_omp_workspace_elems(s,bq,bk,c.threads));
        out.emplace_back(t.o.size());
    }
    // 返回该调用的实际team；scalar/SIMD为单线程，OMP统计属于计时内调用。
    const auto call=[&](std::size_t n) {
        int actual=1;
        if (n==0) attn::blocked_streaming_scalar_fp32(s,t.q.data(),t.k.data(),t.v.data(),bq,bk,ws[n].data(),out[n].data());
        else if (n==1) attn::blocked_streaming_simd_fp32(s,t.q.data(),t.k.data(),t.v.data(),bq,bk,ws[n].data(),out[n].data());
        else attn::blocked_streaming_simd_omp_fp32(s,t.q.data(),t.k.data(),t.v.data(),bq,bk,configs[n].threads,ws[n].data(),out[n].data(),&actual);
        return actual;
    };
    const auto validate=[&](std::size_t n) {
        const auto e=attn::check_contract(out[n].data(),t.o_ref.data(),t.o.size(),{});
        if (!e.pass) throw std::runtime_error("FP64 contract failed");
        return e;
    };
    for (std::size_t n=0;n<configs.size();++n) { call(n); before.push_back(validate(n)); }
    for (int round=0;round<warmup;++round)
        for (std::size_t j=0;j<configs.size();++j) call((j+round)%configs.size());
    for (int round=0;round<reps;++round)
        for (std::size_t order=0;order<configs.size();++order) {
            const auto n=(order+round)%configs.size();
            const auto start=std::chrono::steady_clock::now();
            const int actual=call(n);
            const auto stop=std::chrono::steady_clock::now();
            samples.push_back({n,round,int(order),actual,std::chrono::duration<double,std::milli>(stop-start).count()});
        }
    for (std::size_t n=0;n<configs.size();++n) after.push_back(validate(n));
    if (t.q!=q || t.k!=k || t.v!=v) throw std::runtime_error("input mutation");
    for (auto r:samples) {
        const auto n=r.config;
        std::printf("%zu,%zu,%zu,%s,%d,%d,16,256,%zu,%zu,%zu,20261022,10,20,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,1\n",
            s.L,s.d,s.dv,configs[n].mode,configs[n].threads,r.actual,std::min(bq,s.L),std::min(bk,s.L),
            ws[n].size()*sizeof(float),r.round,r.order,r.ms,before[n].max_abs,before[n].max_rel,after[n].max_abs,after[n].max_rel);
    }
}
}
// 固定实验矩阵；若ISA或数值检查失败，返回非零，上层保留失败状态。
int main() {
    try {
        if (!attn::blocked_streaming_avx2_supported()) throw std::runtime_error("AVX2 unavailable");
        std::puts("L,d,dv,mode,requested_threads,actual_threads,query_block,key_block,effective_bq,effective_bk,workspace_bytes,seed,warmup,reps,sample,order,ms,pre_max_abs,pre_max_rel,post_max_abs,post_max_rel,contract_pass");
        for (auto s:{attn::Shape{128,64,64},{512,64,64},{2048,64,64},{4096,64,64},{512,32,32},{2048,32,32}}) measure(s);
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
