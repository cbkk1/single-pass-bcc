#!/bin/bash
# Build a tests/*.cu driver the same way as tools/build-case.sh.
#   ./tools/build-tests.sh test_dense_pipeline
set -e
cd "$(dirname "$0")/.."
t=${1:?usage: build-tests.sh <test name without .cu>}
nvcc -O3 -std=c++17 -arch=sm_80 -rdc=true --expt-relaxed-constexpr \
     -Xcompiler -fopenmp -Iinclude \
     tests/$t.cu \
     $(find src -name '*.cu' ! -name 'main.cu' ! -name 'extern_bcc.cu') \
     -o build/$t
echo "built build/$t"
