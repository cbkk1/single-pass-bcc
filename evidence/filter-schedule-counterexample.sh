#!/bin/bash
# Minimal counterexample: the per-round filter cannot be deferred.
#
# 17 vertices, 16 edges, streamed ONE EDGE PER BATCH (so the result does not
# depend on where batch boundaries happen to fall).  Shrunk by greedy edge
# deletion from a random failure found by test_dense_pipeline (seed 7).
#
#   core 12-cycle : 0-1-11-12-13-4-8-7-10-9-2-3-0
#   pendants      : 0-6, 4-5, 9-16   plus the disjoint bridge 14-15
#   truth         : 5 BCCs
#
# Compression first fires at round 13 (the edge 13-4 closes the cycle).
# The filter must then run at round 14.  Deferring it by a single round
# loses a BCC.
cd "$(dirname "$0")/.."
E="0 1  2 3  4 5  3 0  6 0  7 8  9 10  7 10  4 8  2 9  11 1  12 13  12 11  13 4  14 15  9 16"

echo "=== filter every round (default) ==="
./build/case 16 17 $E 2>/dev/null | tail -2

echo "=== filter only on the last batch (BCC_FILTER_LAST=1) ==="
BCC_FILTER_LAST=1 ./build/case 16 17 $E 2>/dev/null | tail -5

echo "=== first round at which the filter may start ==="
for f in 13 14 15; do
  printf "  from round %-3s : %s\n" $f \
    "$(BCC_FILTER_FROM=$f ./build/case 16 17 $E 2>/dev/null | grep 'count  got')"
done
