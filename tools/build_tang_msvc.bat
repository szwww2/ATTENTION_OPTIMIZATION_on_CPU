@echo off
REM Run from the repository root in an x64 Visual Studio Developer Command Prompt.
REM All routes use the same MSVC /O2 /arch:AVX2 flags in this exploratory comparison.
if not exist build mkdir build
cl /nologo /std:c++17 /O2 tools\cpu_topology_win.cpp /Fo:build\ /Fe:build\cpu_topology_win.exe
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /O2 /arch:AVX2 /openmp /EHsc /Iinclude ^
  src\oracle_fp64.cpp src\full_matrix_scalar.cpp src\rowwise_scalar.cpp ^
  src\input.cpp src\metrics.cpp src\simd_routes.cpp src\omp_routes.cpp ^
  tools\bench_tang.cpp /Fo:build\ /Fe:build\attn_bench_tang_msvc.exe
exit /b %errorlevel%
