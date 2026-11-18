#!/bin/bash
# Companion to filter-schedule-counterexample.sh: the SAME skipped-round
# pathology, but with the |f^-1(x)| != 1 realness test (BCC_PRED=dis), which
# fails in the opposite direction -- over-merge instead of split.
#
# 12 vertices, 12 edges, ONE EDGE PER BATCH.
#   9-cycle  : 0-1-8-7-6-5-2-3-4-0
#   pendants : 1-9, 6-10, 4-11
#   truth    : 4 BCCs
#
# Compression fires at round 8 (edge 2-5 closes the cycle).  The filter must
# then run at round 10.  Deferring it to round 11 swallows the bridge {4,11}.
cd "$(dirname "$0")/.."
E="0 1  2 3  4 3  4 0  5 6  6 7  8 1  8 7  2 5  9 1  10 6  4 11"

echo "=== filter every round (default) ==="
./build/case 12 12 $E 2>/dev/null | tail -2

echo "=== disjunction predicate, filter only on the last batch ==="
BCC_PRED=dis BCC_FILTER_LAST=1 ./build/case 12 12 $E 2>/dev/null | tail -5

echo "=== first round at which the filter may start ==="
for f in 9 10 11; do
  printf "  from round %-3s : %s\n" $f \
    "$(BCC_PRED=dis BCC_FILTER_FROM=$f ./build/case 12 12 $E 2>/dev/null | grep 'count  got')"
done
