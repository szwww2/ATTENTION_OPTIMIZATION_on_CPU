@echo off
REM Build the scalar baseline and its contract suite (Windows / MSYS2 g++).
REM No auto-vectorization on purpose: this build is the baseline.
setlocal

set CXX=g++
set STD=-std=c++17
set SRC=src\oracle_fp64.cpp src\full_matrix_scalar.cpp src\input.cpp src\metrics.cpp
set BASEFLAGS=%STD% -O3 -fno-tree-vectorize -Wall -Wextra -Iinclude

if not exist build mkdir build

echo [1/1] tests   : attn_tests.exe   (scalar, -fno-tree-vectorize)
%CXX% %BASEFLAGS% %SRC% tests\test_contract.cpp -o build\attn_tests.exe
if errorlevel 1 exit /b 1

echo built: build\attn_tests.exe
endlocal
