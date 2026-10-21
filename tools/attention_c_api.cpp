// Small allocating API for the external-library comparison. No cached workspace.
#include "attn.h"
#include <algorithm>
#include <climits>
#include <new>
#include <omp.h>
#include <vector>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {
// Bounds also keep products and OpenMP block indices safe on supported hosts.
bool valid(int l, int d, int dv, std::size_t nq, std::size_t nv) {
    return l > 0 && l <= 8192 && d > 0 && d <= 4096 && dv > 0 && dv <= 4096
        && nq == std::size_t(l) * d && nv == std::size_t(l) * dv;
}
bool avx2() { return __builtin_cpu_supports("avx2"); }
}

// Return codes: 0 success, 1 invalid arguments, 2 AVX2 unavailable,
// 3 allocation failure, 4 unexpected exception. Never unwind across ctypes.
API int gp8_fill(int l, int d, int dv, unsigned long long seed,
                 float* q, float* k, float* v, std::size_t nq, std::size_t nv) {
    if (!q || !k || !v || !valid(l,d,dv,nq,nv)) return 1;
    attn::fill_uniform(q,nq,seed*3+1,-1,1);
    attn::fill_uniform(k,nq,seed*3+2,-1,1);
    attn::fill_uniform(v,nv,seed*3+3,-1,1);
    return 0;
}

API int gp8_oracle(int l, int d, int dv, const float* q, const float* k,
                   const float* v, double* out, std::size_t nq, std::size_t nv,
                   std::size_t no) {
    if (!q || !k || !v || !out || !valid(l,d,dv,nq,nv) || no != nv) return 1;
    try {
        attn::oracle_fp64({std::size_t(l),std::size_t(d),std::size_t(dv)},q,k,v,out);
        return 0;
    } catch (const std::bad_alloc&) { return 3; } catch (...) { return 4; }
}

API int gp8_threads(int requested) {
    if (requested < 1 || requested > 64) return -1;
    omp_set_dynamic(0);
    int actual = 0;
#pragma omp parallel num_threads(requested)
    {
#pragma omp single
        actual = omp_get_num_threads();
    }
    return actual;
}

// route: 0 full scalar, 1 row scalar, 2 full SIMD, 3 row SIMD,
//        4 full SIMD+OpenMP, 5 row SIMD+OpenMP.
API int gp8_run(int route, int l, int d, int dv, int bq, int threads,
                const float* q, const float* k, const float* v, float* out,
                std::size_t nq, std::size_t nv, std::size_t no) {
    if (!q || !k || !v || !out || !valid(l,d,dv,nq,nv) || no != nv
        || route < 0 || route > 5 || bq < 1 || bq > l || threads < 1
        || threads > 64 || (route < 4 && threads != 1)) return 1;
    if (route >= 2 && !avx2()) return 2;
    const attn::Shape s{std::size_t(l),std::size_t(d),std::size_t(dv)};
    const std::size_t count = route % 2 == 0 ? s.L*s.L
        : std::size_t(route >= 4 ? threads : 1)*bq*s.L;
    if (count > 128ULL*1024*1024) return 1; // 512 MiB workspace limit
    try {
        // Allocation, zero-initialization and release belong to this API call.
        std::vector<float> scratch(count);
        switch (route) {
        case 0: attn::full_matrix_scalar_fp32(s,q,k,v,scratch.data(),out); break;
        case 1: attn::rowwise_scalar_fp32(s,q,k,v,bq,scratch.data(),out); break;
        case 2: attn::full_matrix_simd_fp32(s,q,k,v,scratch.data(),out); break;
        case 3: attn::rowwise_simd_fp32(s,q,k,v,bq,scratch.data(),out); break;
        case 4: attn::full_matrix_simd_omp_fp32(s,q,k,v,bq,threads,scratch.data(),out); break;
        case 5: attn::rowwise_simd_omp_fp32(s,q,k,v,bq,threads,scratch.data(),out); break;
        }
        return 0;
    } catch (const std::bad_alloc&) { return 3; } catch (...) { return 4; }
}
