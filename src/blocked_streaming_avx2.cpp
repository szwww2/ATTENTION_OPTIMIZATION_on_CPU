// AVX2编译单元；点积/加权累加复用Tang f9f088c的共享helper，保持原有tile次序。
#include "blocked_streaming_internal.h"
#include "attn_simd.h"
#include <algorithm>
#include <cmath>

namespace attn::streaming_detail {
namespace {
// 将n个FP32状态乘alpha；指针不要求32字节对齐，余数走标量，无分配。
void rescale(float* u, std::size_t n, float alpha) {
    const __m256 a = _mm256_set1_ps(alpha);
    std::size_t y=0;
    for (; n-y>=8; y+=8)
        _mm256_storeu_ps(u+y, _mm256_mul_ps(_mm256_loadu_ps(u+y),a));
    for (; y<n; ++y) u[y]*=alpha;
}
// 将累加和逐元素除以分母并覆盖输出；用真实除法而非近似倒数，支持尾部。
void normalize(const float* u, float* out, std::size_t n, float l) {
    const __m256 denominator = _mm256_set1_ps(l);
    std::size_t y=0;
    for (; n-y>=8; y+=8)
        _mm256_storeu_ps(out+y, _mm256_div_ps(_mm256_loadu_ps(u+y),denominator));
    for (; y<n; ++y) out[y]=u[y]/l;
}
}

// 处理[i0,min(i0+bq,L))；K/V仍按key行→query行顺序复用。exp/分母归约保持标量。
void avx2_tile(const Shape& s, const float* q, const float* k, const float* v,
               std::size_t bq, std::size_t bk, std::size_t i0,
               float* workspace, float* out) {
    float* score=workspace;
    float* maximum=score+bq*bk;
    float* normalizer=maximum+bq;
    float* accum=normalizer+bq;
    const auto nq=std::min(bq,s.L-i0);
    const float scale=1.0f/std::sqrt(static_cast<float>(s.d));
    std::fill(accum,accum+nq*s.dv,0.0f);
    std::fill(normalizer,normalizer+nq,0.0f);
    for (std::size_t j0=0;j0<s.L;) {
        const auto nk=std::min(bk,s.L-j0);
        for (std::size_t j=0;j<nk;++j)
            for (std::size_t r=0;r<nq;++r)
                score[r*bk+j]=dot_avx2_fp32(q+(i0+r)*s.d,k+(j0+j)*s.d,s.d)*scale;

        // 每query维护m/l/u；首块不读旧m，后续m变化同时缩放旧l和u。
        for (std::size_t r=0;r<nq;++r) {
            float* row=score+r*bk;
            float new_max=row[0];
            for (std::size_t j=1;j<nk;++j) new_max=std::max(new_max,row[j]);
            if (j0!=0) {
                new_max=std::max(maximum[r],new_max);
                const float alpha=std::exp(maximum[r]-new_max);
                normalizer[r]*=alpha;
                rescale(accum+r*s.dv,s.dv,alpha);
            }
            maximum[r]=new_max;
            float sum=0.0f;
            for (std::size_t j=0;j<nk;++j) {
                row[j]=std::exp(row[j]-new_max);
                sum+=row[j];
            }
            normalizer[r]+=sum;
        }
        for (std::size_t j=0;j<nk;++j)
            for (std::size_t r=0;r<nq;++r)
                weighted_add_avx2_fp32(score[r*bk+j],v+(j0+j)*s.dv,accum+r*s.dv,s.dv);
        j0+=nk;
    }
    for (std::size_t r=0;r<nq;++r)
        normalize(accum+r*s.dv,out+(i0+r)*s.dv,s.dv,normalizer[r]);
}
}
