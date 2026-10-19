// Two scalar routes, identical inputs, validation before timing, and raw samples.
#include "attn.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Options {
    std::vector<std::size_t> lengths{128, 512};
    std::size_t d = 64, dv = 64, query_block = 64;
    std::uint64_t seed = 20261020;
    int warmup = 10, reps = 20;
    std::string out = "results/scalar";
    bool csv = false;
};

std::uint64_t number(const std::string& text, std::uint64_t low, std::uint64_t high) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected an unsigned decimal integer: " + text);
    const auto value = std::stoull(text);
    if (value < low || value > high) throw std::runtime_error("Value out of range: " + text);
    return value;
}

void usage() {
    std::cout << "attn_bench [--L 1..4096] [--d 1..512] [--dv 1..512]\n"
              << "           [--query-block 1..4096] [--warmup 1..1000] [--reps 3..1000]\n"
              << "           [--seed integer] [--out results/run1] [--csv]\n"
              << "Default: L=128,512; d=dv=64; both routes; 10 warmups, 20 samples.\n"
              << "Writes <out>_raw.csv and <out>_summary.csv without overwriting.\n";
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--csv") { o.csv = true; continue; }
        if (arg != "--L" && arg != "--d" && arg != "--dv" && arg != "--query-block" &&
            arg != "--warmup" && arg != "--reps" && arg != "--seed" && arg != "--out")
            throw std::runtime_error("Unknown option: " + arg);
        if (++i == argc) throw std::runtime_error("Missing value for " + arg);
        const std::string value = argv[i];
        if (arg == "--L") o.lengths = {static_cast<std::size_t>(number(value, 1, 4096))};
        else if (arg == "--d") o.d = static_cast<std::size_t>(number(value, 1, 512));
        else if (arg == "--dv") o.dv = static_cast<std::size_t>(number(value, 1, 512));
        else if (arg == "--query-block") o.query_block = static_cast<std::size_t>(number(value, 1, 4096));
        else if (arg == "--warmup") o.warmup = static_cast<int>(number(value, 1, 1000));
        else if (arg == "--reps") o.reps = static_cast<int>(number(value, 3, 1000));
        else if (arg == "--seed") o.seed = number(value, 0, std::numeric_limits<std::uint64_t>::max());
        else { if (value.empty()) throw std::runtime_error("Empty output prefix"); o.out = value; }
    }
    return o;
}

struct Route {
    const char* name;
    bool rowwise;
    std::vector<float> workspace, output;
    std::vector<double> times;
    attn::ErrorReport before, after;
};

void call(Route& r, const attn::Shape& s, const attn::Tensors& t, std::size_t bq) {
    if (r.rowwise)
        attn::rowwise_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(), bq,
                                  r.workspace.data(), r.output.data());
    else
        attn::full_matrix_scalar_fp32(s, t.q.data(), t.k.data(), t.v.data(),
                                     r.workspace.data(), r.output.data());
}

attn::ErrorReport validate(const Route& r, const attn::Tensors& t, const char* phase) {
    const auto error = attn::check_contract(r.output.data(), t.o_ref.data(), r.output.size(), {});
    if (!error.pass)
        throw std::runtime_error(std::string(r.name) + " failed the fixed numerical contract " + phase);
    return error;
}

// Inclusive interpolation at p*(n-1), matching Python's inclusive quantiles.
double quantile(const std::vector<double>& sorted, double p) {
    const double position = p * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(position);
    const auto hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (position - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

struct ShapeResult {
    attn::Shape shape;
    std::size_t bq;
    std::vector<Route> routes;
};

ShapeResult measure(const Options& o, std::size_t L) {
    ShapeResult result{{L, o.d, o.dv}, std::min(o.query_block, L), {}};
    const auto& s = result.shape;
    attn::Tensors t;
    t.resize(s);
    t.fill_qkv(o.seed);  // Generated once; both routes read these exact buffers.
    attn::oracle_fp64(s, t.q.data(), t.k.data(), t.v.data(), t.o_ref.data());
    result.routes = {{"full_matrix", false, {}, {}, {}, {}, {}},
                     {"rowwise", true, {}, {}, {}, {}, {}}};
    for (auto& r : result.routes) {
        const auto elems = r.rowwise ? attn::rowwise_workspace_elems(s, result.bq)
                                     : attn::full_matrix_workspace_elems(s);
        r.workspace.assign(elems, 0.0f);
        r.output.assign(attn::v_elems(s), std::numeric_limits<float>::quiet_NaN());
        r.times.reserve(static_cast<std::size_t>(o.reps));
        call(r, s, t, result.bq);
        r.before = validate(r, t, "before timing");  // Both routes pass before any samples.
    }
    for (int round = 0; round < o.warmup; ++round)
        for (int slot = 0; slot < 2; ++slot)
            call(result.routes[static_cast<std::size_t>((round + slot) % 2)], s, t, result.bq);
    for (int round = 0; round < o.reps; ++round) {
        // Alternate which route goes first, instead of always favouring one route.
        for (int slot = 0; slot < 2; ++slot) {
            auto& r = result.routes[static_cast<std::size_t>((round + slot) % 2)];
            const auto start = std::chrono::steady_clock::now();
            call(r, s, t, result.bq);
            const auto end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (!(ms > 0.0) || !std::isfinite(ms)) throw std::runtime_error("Invalid timer sample");
            r.times.push_back(ms);
        }
    }
    for (auto& r : result.routes) r.after = validate(r, t, "after timing");
    return result;
}

void write_results(const Options& o, const std::vector<ShapeResult>& results) {
    std::ofstream raw(o.out + "_raw.csv"), summary(o.out + "_summary.csv");
    if (!raw || !summary) throw std::runtime_error("Cannot open CSV output files");
    raw << std::setprecision(17);
    summary << std::setprecision(17);
    const std::string header = "route,L,d,dv,seed,query_block,warmup,reps,median_ms,min_ms,q1_ms,q3_ms,iqr_ms,"
                               "gflops,workspace_bytes,pre_max_abs,pre_max_rel,post_max_abs,post_max_rel,contract_pass\n";
    raw << "route,L,d,dv,seed,query_block,sample,order,ms\n";
    summary << header;
    if (o.csv) std::cout << std::setprecision(17) << header;
    else std::cout << "route L median_ms IQR_ms workspace_bytes contract\n";
    for (const auto& result : results) {
        const auto& s = result.shape;
        for (std::size_t index = 0; index < result.routes.size(); ++index) {
            const auto& r = result.routes[index];
            const auto block = r.rowwise ? result.bq : 0;
            for (std::size_t sample = 0; sample < r.times.size(); ++sample)
                raw << r.name << ',' << s.L << ',' << s.d << ',' << s.dv << ',' << o.seed << ','
                    << block << ',' << sample << ',' << ((index + sample) % 2) << ',' << r.times[sample] << '\n';
            auto sorted = r.times;
            std::sort(sorted.begin(), sorted.end());
            const double median = quantile(sorted, 0.5), q1 = quantile(sorted, 0.25), q3 = quantile(sorted, 0.75);
            const double flops = 2.0 * static_cast<double>(s.L) * static_cast<double>(s.L) *
                                 static_cast<double>(s.d + s.dv);
            const auto bytes = r.workspace.size() * sizeof(float);
            const auto row = [&](std::ostream& out) {
                out << r.name << ',' << s.L << ',' << s.d << ',' << s.dv << ',' << o.seed << ',' << block << ','
                    << o.warmup << ',' << o.reps << ',' << median << ',' << sorted.front() << ',' << q1 << ',' << q3
                    << ',' << q3-q1 << ',' << flops/(median*1e6) << ',' << bytes << ',' << r.before.max_abs << ','
                    << r.before.max_rel << ',' << r.after.max_abs << ',' << r.after.max_rel << ",1\n";
            };
            row(summary);
            if (o.csv) row(std::cout);
            else std::cout << r.name << ' ' << s.L << ' ' << std::fixed << std::setprecision(4)
                           << median << ' ' << q3-q1 << ' ' << bytes << " PASS\n";
        }
    }
    raw.close(); summary.close();
    if (!raw || !summary) throw std::runtime_error("Could not finish writing CSV files");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
        const Options o = parse(argc, argv);
        if (std::filesystem::exists(o.out + "_raw.csv") || std::filesystem::exists(o.out + "_summary.csv"))
            throw std::runtime_error("Output already exists; choose a new --out prefix");
        const auto parent = std::filesystem::path(o.out).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::vector<ShapeResult> results;
        for (const auto L : o.lengths) results.push_back(measure(o, L));
        // No performance CSV is written unless every requested shape passes both checks.
        write_results(o, results);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
