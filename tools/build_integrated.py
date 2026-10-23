"""Build the shared scalar/SIMD/OpenMP benchmark and tests with GCC."""
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
OBJECTS = BUILD / "integrated_objects"
CXX = os.environ.get("CXX", "g++")
BASE_FLAGS = [
    "-std=c++17", "-O3", "-fno-tree-vectorize", "-ffp-contract=off",
    "-Wall", "-Wextra", "-Iinclude",
]
SIMD_FLAGS = ["-mavx2"]
OMP_FLAGS = ["-fopenmp"]
SUFFIX = ".exe" if os.name == "nt" else ""


def compile_source(source, extra_flags=()):
    """Compile one source; only explicit SIMD sources receive AVX2 flags."""
    obj = OBJECTS / (Path(source).stem + ".o")
    print("Compile", source, flush=True)
    subprocess.run(
        [CXX, *BASE_FLAGS, *extra_flags, "-c", source,
         "-o", str(obj.relative_to(ROOT))],
        cwd=ROOT, check=True,
    )
    return obj.relative_to(ROOT)


def link_program(name, objects, openmp=False):
    print("Link", name + SUFFIX, flush=True)
    flags = OMP_FLAGS if openmp else []
    subprocess.run(
        [CXX, *map(str, objects), *flags, "-o", "build/" + name + SUFFIX],
        cwd=ROOT, check=True,
    )


def main():
    OBJECTS.mkdir(parents=True, exist_ok=True)
    record = BUILD / "integrated_build.txt"
    # A failed rebuild must not leave a successful build record behind.
    record.write_text("status=building\n", encoding="utf-8")
    try:
        version = subprocess.check_output(
            [CXX, "--version"], cwd=ROOT, text=True,
            encoding="utf-8", errors="replace",
        ).splitlines()[0]
        core = [compile_source("src/" + name + ".cpp") for name in (
            "oracle_fp64", "full_matrix_scalar", "rowwise_scalar", "input", "metrics",
            "blocked_streaming_scalar",
        )]
        simd = compile_source("src/simd_routes.cpp", SIMD_FLAGS)
        omp = compile_source("src/omp_routes.cpp", OMP_FLAGS)
        streaming_simd = compile_source("src/blocked_streaming_avx2.cpp", SIMD_FLAGS)
        streaming_dispatch = compile_source("src/blocked_streaming_optimized.cpp", OMP_FLAGS)
        streaming_objects = [streaming_simd, streaming_dispatch]

        scalar_test = compile_source("tests/test_contract.cpp")
        link_program("attn_tests", [*core, scalar_test])
        block_test = compile_source("tests/test_blocks.cpp")
        link_program("attn_blocks", [*core, block_test])

        primitive_test = compile_source("tests/test_simd_kernels.cpp", SIMD_FLAGS)
        link_program("attn_simd_kernels_tests", [primitive_test])
        simd_test = compile_source("tests/test_simd_contract.cpp", ["-DATTN_TEST_SIMD"])
        link_program("attn_simd_tests", [*core, simd, simd_test])
        omp_test = compile_source("tests/test_omp_routes.cpp", OMP_FLAGS)
        link_program("attn_omp_tests", [*core, simd, omp, omp_test], openmp=True)

        streaming_test = compile_source("tests/test_streaming.cpp")
        link_program("attn_streaming_tests", [*core, streaming_test])
        streaming_opt_test = compile_source("tests/test_streaming_optimized.cpp", OMP_FLAGS)
        link_program("attn_streaming_optimized_tests",
                     [*core, *streaming_objects, streaming_opt_test], openmp=True)

        benchmark = compile_source(
            "tools/bench_scalar.cpp", ["-DATTN_WITH_SIMD", *OMP_FLAGS],
        )
        link_program("attn_bench_all", [*core, simd, omp, *streaming_objects, benchmark], openmp=True)
        record.write_text(
            "status=passed\n"
            + "compiler=" + version + "\n"
            + "base_flags=" + " ".join(BASE_FLAGS) + "\n"
            + "simd_extra_flags=" + " ".join(SIMD_FLAGS) + "\n"
            + "openmp_extra_flags=" + " ".join(OMP_FLAGS) + "\n"
            + "benchmark_defines=ATTN_WITH_SIMD\n"
            + "simd_test_defines=ATTN_TEST_SIMD\n"
            + "scalar_translation_units=baseline flags; no AVX2\n"
            + "avx2_translation_units=src/simd_routes.cpp src/blocked_streaming_avx2.cpp\n",
            encoding="utf-8",
        )
        print("Built integrated benchmark and seven test executables.")
        return 0
    except (OSError, subprocess.CalledProcessError) as error:
        record.write_text("status=failed\n", encoding="utf-8")
        print("Integrated build failed:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
