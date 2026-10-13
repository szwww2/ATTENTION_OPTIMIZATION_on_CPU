// Pass rule from the proposal.
#include "attn.h"

#include <cmath>
#include <limits>

namespace attn {

ErrorReport check_contract(const float* got, const double* ref, std::size_t n, const Contract& c) {
    ErrorReport r;
    r.worst_margin = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < n; ++i) {
        const double g = static_cast<double>(got[i]);
        const double e = ref[i];
        const bool g_ok = std::isfinite(g);
        const bool e_ok = std::isfinite(e);
        if (!g_ok || !e_ok) {
            ++r.non_finite;
            continue;
        }
        const double diff = std::fabs(g - e);
        const double denom = std::fmax(c.rel_denom_floor, std::fabs(e));
        const double rel = diff / denom;
        const double margin = diff - (c.atol + c.rtol * std::fabs(e));

        if (diff > r.max_abs) r.max_abs = diff;
        if (rel > r.max_rel) r.max_rel = rel;
        if (margin > r.worst_margin) {
            r.worst_margin = margin;
            r.worst_index = i;
        }
    }

    if (r.non_finite == 0 && r.worst_margin <= 0.0) r.pass = true;
    if (r.non_finite > 0) r.pass = false;
    return r;
}

}  // namespace attn
