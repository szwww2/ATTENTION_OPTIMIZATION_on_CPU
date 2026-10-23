// baseline ISA编译，先检查AVX2；不让不支持ISA的CPU进入AVX2编译单元。
#include "blocked_streaming_internal.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <omp.h>

namespace attn {
// GCC/Clang x86内建检查含OS AVX状态支持；其它架构/编译器明确返回false。
bool blocked_streaming_avx2_supported() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
namespace {
// 有效缓冲区/ISA前提检查，不读写输入输出；调用前必须按helper预分配且不别名。
void validate(const float* q,const float* k,const float* v,float* ws,float* out) {
    if (!q || !k || !v || !ws || !out)
        throw std::invalid_argument("streaming received a null data buffer");
    if (!blocked_streaming_avx2_supported())
        throw std::runtime_error("streaming SIMD requires AVX2 CPU/OS support");
}
}
// 独立SIMD入口；每块复用同一scratch，尺寸与scalar helper一致。
void blocked_streaming_simd_fp32(const Shape& s,const float* q,const float* k,const float* v,
                                 std::size_t bq,std::size_t bk,float* ws,float* out) {
    blocked_streaming_workspace_elems(s,bq,bk);
    validate(q,k,v,ws,out);
    bq=std::min(s.L,std::max(std::size_t{1},bq));
    bk=std::min(s.L,std::max(std::size_t{1},bk));
    for (std::size_t i0=0;i0<s.L;i0+=std::min(bq,s.L-i0))
        streaming_detail::avx2_tile(s,q,k,v,bq,bk,i0,ws,out);
}
// 返回请求线程数乘单线程完整scratch；不依赖OMP_NUM_THREADS/动态team大小。
std::size_t blocked_streaming_omp_workspace_elems(const Shape& s,std::size_t bq,
                                                 std::size_t bk,int threads) {
    if (threads<=0) throw std::invalid_argument("streaming requires positive threads");
    const auto one=blocked_streaming_workspace_elems(s,bq,bk);
    const auto count=static_cast<std::size_t>(threads);
    if (one>std::numeric_limits<std::size_t>::max()/sizeof(float)/count)
        throw std::overflow_error("streaming threaded workspace overflow");
    return one*count;
}
// 每线程处理静态query块集合，独占ws[tid*S:(tid+1)*S]，输出query行互不重叠。
// 不拆key归约，保证同一SIMD核跨线程数的运算顺序相同；运行时可缩小team。
void blocked_streaming_simd_omp_fp32(const Shape& s,const float* q,const float* k,const float* v,
                                     std::size_t bq,std::size_t bk,int threads,
                                     float* ws,float* out,int* actual_threads) {
    const auto total=blocked_streaming_omp_workspace_elems(s,bq,bk,threads);
    validate(q,k,v,ws,out);
    const auto stride=total/static_cast<std::size_t>(threads);
    bq=std::min(s.L,std::max(std::size_t{1},bq));
    bk=std::min(s.L,std::max(std::size_t{1},bk));
    const auto blocks=s.L/bq+(s.L%bq!=0);
    // layout已检查L*d字节计数，故blocks<=L<=SIZE_MAX/4，转ptrdiff_t安全。
    const auto end=static_cast<std::ptrdiff_t>(blocks);
    #pragma omp parallel num_threads(threads) default(none) shared(s,q,k,v,bq,bk,ws,out,stride,end,actual_threads)
    {
        const int tid=omp_get_thread_num();
        if (tid==0 && actual_threads) *actual_threads=omp_get_num_threads();
        float* private_ws=ws+static_cast<std::size_t>(tid)*stride;
        #pragma omp for schedule(static)
        for (std::ptrdiff_t block=0;block<end;++block)
            streaming_detail::avx2_tile(s,q,k,v,bq,bk,static_cast<std::size_t>(block)*bq,private_ws,out);
    }
}
}
