#include "attn.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace {
int checks=0, failures=0, largest_team=0;
double max_abs=0;
// 累计断言，失败打印名称；避免NDEBUG导致测试被移除。
void check(bool ok,const char* label) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr,"FAIL %s\n",label); }
}
// 两端guard和+1 float偏移，覆盖非32字节对齐输入、scratch与输出；n为有效元素数。
struct Guarded {
    std::vector<float> data;
    explicit Guarded(std::size_t n):data(n+2,123456.0f) {
        std::fill(data.begin()+1,data.end()-1,std::numeric_limits<float>::quiet_NaN());
    }
    float* ptr() { return data.data()+1; }
    bool intact() const { return data.front()==123456.0f && data.back()==123456.0f; }
};
// 使用独立FP64契约，而非与scalar的逐位相等；返回报告供错误统计。
void validate(const Guarded& got,const attn::Tensors& t) {
    const auto e=attn::check_contract(got.data.data()+1,t.o_ref.data(),t.o_ref.size(),{});
    max_abs=std::max(max_abs,e.max_abs);
    check(e.pass,"FP64 contract");
    check(got.intact(),"output guards");
}
// 同输入跨tile/SIMD/OpenMP与线程数验证；kind=1等分数，2大分数，3递增最大值。
void exercise(attn::Shape s,int seed,int kind=0) {
    attn::Tensors t; t.resize(s); t.fill_qkv(seed);
    if (kind==1) std::fill(t.q.begin(),t.q.end(),0.0f);
    if (kind==2) for (auto& x:t.q) x*=100.0f;
    if (kind==3) {
        std::fill(t.q.begin(),t.q.end(),1.0f);
        for (std::size_t j=0;j<s.L;++j) t.k[j]=float(2*j)-8;
    }
    attn::oracle_fp64(s,t.q.data(),t.k.data(),t.v.data(),t.o_ref.data());
    Guarded q(t.q.size()),k(t.k.size()),v(t.v.size());
    std::copy(t.q.begin(),t.q.end(),q.ptr());
    std::copy(t.k.begin(),t.k.end(),k.ptr());
    std::copy(t.v.begin(),t.v.end(),v.ptr());
    for (std::size_t bq:{std::size_t{0},std::size_t{3},std::size_t{16},s.L+1})
        for (std::size_t bk:{std::size_t{0},std::size_t{7},std::size_t{256},s.L+1}) {
            const auto stride=attn::blocked_streaming_workspace_elems(s,bq,bk);
            Guarded ws(stride),out(t.o.size());
            attn::blocked_streaming_simd_fp32(s,q.ptr(),k.ptr(),v.ptr(),bq,bk,ws.ptr(),out.ptr());
            validate(out,t); check(ws.intact(),"SIMD scratch guards");
            const auto expected=out.data;
            attn::blocked_streaming_simd_fp32(s,q.ptr(),k.ptr(),v.ptr(),bq,bk,ws.ptr(),out.ptr());
            check(out.data==expected,"SIMD repeated overwrite");
            for (int threads:{1,2,3,4,8}) {
                const auto size=attn::blocked_streaming_omp_workspace_elems(s,bq,bk,threads);
                check(size==stride*std::size_t(threads),"total thread scratch");
                Guarded multi(size),result(t.o.size());
                int actual=0;
                attn::blocked_streaming_simd_omp_fp32(s,q.ptr(),k.ptr(),v.ptr(),bq,bk,
                                                     threads,multi.ptr(),result.ptr(),&actual);
                check(actual>0 && actual<=threads,"actual team bound");
                largest_team=std::max(largest_team,actual);
                validate(result,t);
                check(result.data==expected,"thread count does not change SIMD arithmetic");
                check(multi.intact(),"OMP scratch guards");
                // 重复调用使用脏scratch/out，必须重置状态且不依赖上一team布局。
                attn::blocked_streaming_simd_omp_fp32(s,q.ptr(),k.ptr(),v.ptr(),bq,bk,
                                                     threads,multi.ptr(),result.ptr());
                check(result.data==expected,"OMP repeated overwrite");
            }
        }
    check(std::equal(t.q.begin(),t.q.end(),q.ptr()) && q.intact(),"Q immutable");
    check(std::equal(t.k.begin(),t.k.end(),k.ptr()) && k.intact(),"K immutable");
    check(std::equal(t.v.begin(),t.v.end(),v.ptr()) && v.intact(),"V immutable");
}
// 非法参数必须抛指定异常；检查发生在线程启动之前，输出不应改变。
template<class E,class F> void rejects(F f) {
    bool caught=false;
    try { f(); } catch (const E&) { caught=true; }
    check(caught,"invalid argument rejected");
}
// 同时从两个外层线程调用，nested team序列化；保证各调用独立且不改全局设置。
void nested_calls() {
    attn::Shape s{33,31,33};
    attn::Tensors t; t.resize(s); t.fill_qkv(7);
    attn::oracle_fp64(s,t.q.data(),t.k.data(),t.v.data(),t.o_ref.data());
    const auto n=attn::blocked_streaming_omp_workspace_elems(s,3,7,4);
    Guarded ws0(n),ws1(n),out0(t.o.size()),out1(t.o.size());
    int a=0,b=0;
    const int levels=omp_get_max_active_levels();
    omp_set_max_active_levels(1);
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        attn::blocked_streaming_simd_omp_fp32(s,t.q.data(),t.k.data(),t.v.data(),3,7,4,ws0.ptr(),out0.ptr(),&a);
        #pragma omp section
        attn::blocked_streaming_simd_omp_fp32(s,t.q.data(),t.k.data(),t.v.data(),3,7,4,ws1.ptr(),out1.ptr(),&b);
    }
    check(a>=1 && b>=1 && a<=4 && b<=4,"nested actual teams");
    validate(out0,t); validate(out1,t);
    check(ws0.intact() && ws1.intact(),"concurrent caller guards");
    check(out0.data==out1.data,"concurrent caller results");
    check(omp_get_max_active_levels()==1,"kernel preserves runtime setting");
    omp_set_max_active_levels(levels);
}
}

// 完整测试入口；无AVX2明确skip(77)，不能冒称优化测试通过。
int main() {
    if (!attn::blocked_streaming_avx2_supported()) { std::puts("SKIP: AVX2 unavailable"); return 77; }
    for (auto s:{attn::Shape{1,1,1},{7,7,9},{17,8,8},{127,31,33},{129,65,64},{256,64,33}})
        for (int seed:{1,2,3}) exercise(s,seed);
    exercise({17,31,33},1,1); exercise({33,64,33},1,2); exercise({9,1,3},1,3);
    float cell=42;
    for (int threads:{0,-1}) rejects<std::invalid_argument>([&]{
        attn::blocked_streaming_simd_omp_fp32({1,1,1},&cell,&cell,&cell,1,1,threads,&cell,&cell);
    });
    rejects<std::invalid_argument>([&]{attn::blocked_streaming_simd_fp32({1,1,1},nullptr,&cell,&cell,1,1,&cell,&cell);});
    rejects<std::invalid_argument>([]{attn::blocked_streaming_omp_workspace_elems({0,1,1},1,1,2);});
    rejects<std::overflow_error>([]{attn::blocked_streaming_omp_workspace_elems(
        {1,1,std::numeric_limits<std::size_t>::max()/sizeof(float)/2},1,1,4);});
    check(cell==42,"invalid calls leave output unchanged");
    nested_calls();
    std::printf("optimized streaming: %d checks, %d failures, max_abs=%.9g, largest_actual_team=%d\n",
                checks,failures,max_abs,largest_team);
    return failures?1:0;
}
