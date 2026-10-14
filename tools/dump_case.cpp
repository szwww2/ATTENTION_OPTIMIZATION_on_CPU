// Dump one case (q/k/v, reference, FP32) for crosscheck_numpy.py.
#include "attn.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void dump_floats(const std::string& path, const std::vector<float>& v) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(2);
    }
    for (float x : v) std::fprintf(f, "%.9g\n", static_cast<double>(x));
    std::fclose(f);
}

void dump_doubles(const std::string& path, const std::vector<double>& v) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(2);
    }
    // %.17g round-trips a double
    for (double x : v) std::fprintf(f, "%.17g\n", x);
    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <L> <d> <dv> <seed> <out_prefix>\n", argv[0]);
        return 2;
    }
    attn::Shape s{static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)),
                  static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)),
                  static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10))};
    const std::uint64_t seed = std::strtoull(argv[4], nullptr, 10);
    const std::string prefix = argv[5];

    attn::Tensors t;
    t.resize(s);
    t.fill_device_neutral(seed);
    attn::oracle_fp64(s, t.q.data(), t.k.data(), t.v.data(), t.o_ref.data());
    attn::full_matrix_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), t.ws.data(), t.o.data());

    dump_floats(prefix + "_q.txt", t.q);
    dump_floats(prefix + "_k.txt", t.k);
    dump_floats(prefix + "_v.txt", t.v);
    dump_doubles(prefix + "_oref.txt", t.o_ref);
    dump_floats(prefix + "_o32.txt", t.o);

    const attn::ErrorReport r =
        attn::check_contract(t.o.data(), t.o_ref.data(), t.o.size(), attn::Contract{});
    std::printf("dumped L=%llu d=%llu dv=%llu seed=%llu -> %s_*.txt (C++ self-check: max_abs=%.3e %s)\n",
                static_cast<unsigned long long>(s.L), static_cast<unsigned long long>(s.d),
                static_cast<unsigned long long>(s.dv), static_cast<unsigned long long>(seed),
                prefix.c_str(), r.max_abs, r.pass ? "PASS" : "FAIL");
    return r.pass ? 0 : 1;
}
