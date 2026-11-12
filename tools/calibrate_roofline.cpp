// Measured reference workloads, not guaranteed hardware peaks.
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;

__attribute__((target("avx2"),noinline))
double arithmetic(int iterations) {
    __m256 a0=_mm256_set1_ps(.1f), a1=_mm256_set1_ps(.2f), a2=_mm256_set1_ps(.3f), a3=_mm256_set1_ps(.4f);
    __m256 a4=_mm256_set1_ps(.5f), a5=_mm256_set1_ps(.6f), a6=_mm256_set1_ps(.7f), a7=_mm256_set1_ps(.8f);
    const __m256 m=_mm256_set1_ps(.99999994f), b=_mm256_set1_ps(.0000001f);
    for(int i=0;i<iterations;++i) {
        a0=_mm256_add_ps(_mm256_mul_ps(a0,m),b); a1=_mm256_add_ps(_mm256_mul_ps(a1,m),b);
        a2=_mm256_add_ps(_mm256_mul_ps(a2,m),b); a3=_mm256_add_ps(_mm256_mul_ps(a3,m),b);
        a4=_mm256_add_ps(_mm256_mul_ps(a4,m),b); a5=_mm256_add_ps(_mm256_mul_ps(a5,m),b);
        a6=_mm256_add_ps(_mm256_mul_ps(a6,m),b); a7=_mm256_add_ps(_mm256_mul_ps(a7,m),b);
    }
    alignas(32) float lanes[8];
    const __m256 sum=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)),
                                 _mm256_add_ps(_mm256_add_ps(a4,a5),_mm256_add_ps(a6,a7)));
    _mm256_store_ps(lanes,sum);
    double result=0; for(float x:lanes) result+=x;
    return result;
}

__attribute__((target("avx2"),noinline))
double read_stream(const float* data,std::size_t n,int passes) {
    __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    for(int pass=0;pass<passes;++pass) for(std::size_t i=0;i<n;i+=64) {
        a0=_mm256_add_ps(a0,_mm256_loadu_ps(data+i));
        a1=_mm256_add_ps(a1,_mm256_loadu_ps(data+i+8));
        a2=_mm256_add_ps(a2,_mm256_loadu_ps(data+i+16));
        a3=_mm256_add_ps(a3,_mm256_loadu_ps(data+i+24));
        a4=_mm256_add_ps(a4,_mm256_loadu_ps(data+i+32));
        a5=_mm256_add_ps(a5,_mm256_loadu_ps(data+i+40));
        a6=_mm256_add_ps(a6,_mm256_loadu_ps(data+i+48));
        a7=_mm256_add_ps(a7,_mm256_loadu_ps(data+i+56));
    }
    alignas(32) float lanes[8];
    const __m256 sum=_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3)),
                                 _mm256_add_ps(_mm256_add_ps(a4,a5),_mm256_add_ps(a6,a7)));
    _mm256_store_ps(lanes,sum);
    double result=0; for(float x:lanes) result+=x;
    return result;
}

int main(int argc,char** argv) {
    if(argc!=2) return 1;
    const int threads=std::atoi(argv[1]);
    if(threads!=1 && threads!=2 && threads!=4 && threads!=8) return 1;
    if(!__builtin_cpu_supports("avx2")) { std::cerr<<"AVX2 unavailable\n"; return 2; }
    for (int count : {0,1,17}) {
        float values[8]={.1f,.2f,.3f,.4f,.5f,.6f,.7f,.8f};
        for(int i=0;i<count;++i) for(float& x:values) x=x*.99999994f+.0000001f;
        double expected=0; for(float x:values) expected+=8*double(x);
        if(std::abs(arithmetic(count)-expected)>1e-6*expected) {
            std::cerr<<"Arithmetic self-check failed\n"; return 5;
        }
    }
    omp_set_dynamic(0);
    constexpr std::size_t bytes=512ULL*1024*1024, elements=bytes/sizeof(float);
    constexpr int iterations=10000000, passes=4, repetitions=7;
    std::vector<float> data(elements,1.f); // allocation/initialization outside samples
    std::cout<<"workload,threads,actual_threads,sample,ms,work_units,units_per_second,checksum\n"<<std::setprecision(17);
    // Rotate the two workloads; warmup samples are excluded from the CSV.
    for(int sample=-2;sample<repetitions;++sample) for(int position=0;position<2;++position) {
        const bool read=((sample+2+position)%2)==1;
        double checksum=0;
        int actual=0;
        const auto start=Clock::now();
#pragma omp parallel num_threads(threads) reduction(+:checksum)
        {
#pragma omp single
            actual=omp_get_num_threads();
            const int tid=omp_get_thread_num();
            if(read) checksum+=read_stream(data.data()+elements/actual*tid,elements/actual,passes);
            else checksum+=arithmetic(iterations);
        }
        const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        if(actual!=threads || !std::isfinite(checksum) || !(checksum>0) || !(ms>0)) return 3;
        if(read && checksum!=double(elements)*passes) { std::cerr<<"Read checksum failed\n"; return 4; }
        const double work=read ? double(bytes)*passes : double(iterations)*8*16*threads;
        if(sample>=0) std::cout<<(read?"read_stream":"avx2_mul_add")<<','<<threads<<','<<actual<<','<<sample<<','
                              <<ms<<','<<work<<','<<work/(ms/1000)<<','<<checksum<<'\n';
    }
}
