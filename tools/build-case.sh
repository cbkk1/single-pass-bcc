#!/bin/bash
# Build the single-case driver (tools/case.cu).  Not a CMake target: it links
# every src/*.cu except main.cu.
set -e
cd "$(dirname "$0")/.."
nvcc -O3 -std=c++17 -arch=sm_80 -rdc=true --expt-relaxed-constexpr \
     -Xcompiler -fopenmp -Iinclude \
     tools/case.cu \
     $(find src -name '*.cu' ! -name 'main.cu' ! -name 'extern_bcc.cu') \
     -o build/case
echo "built build/case"
