#pragma once
#include "attn.h"

namespace attn::streaming_detail {
// 内部单query块AVX2计算；调用者已验证尺寸/指针/ISA，bq,bk为有效块长。
// i0<L；写私有scratch与该query块输出，不抛异常，不启动线程。
void avx2_tile(const Shape& s, const float* q, const float* k, const float* v,
               std::size_t bq, std::size_t bk, std::size_t i0,
               float* workspace, float* out);
}
