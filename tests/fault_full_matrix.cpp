// Test-only link wrapper. Never linked by build.bat or build.sh.
#include "attn.h"
#include <cstdlib>
#include <limits>
#include <string>

namespace attn {
// The real implementation is compiled under this name for the fault test only.
void full_matrix_unmodified(const Shape&, const float*, const float*, const float*,
                            float*, float*);

void full_matrix_scalar_fp32(const Shape& s, const float* q, const float* k, const float* v,
                             float* workspace, float* out) {
    full_matrix_unmodified(s, q, k, v, workspace, out);
    static unsigned calls = 0;
    ++calls;
    const char* at = std::getenv("ATTN_TEST_FAIL_AT");
    if (at && calls == std::strtoul(at, nullptr, 10)) {
        const char* mode = std::getenv("ATTN_TEST_FAILURE");
        out[0] = mode && std::string(mode) == "nan"
            ? std::numeric_limits<float>::quiet_NaN() : out[0] + 1.0f;
    }
}
}  // namespace attn
