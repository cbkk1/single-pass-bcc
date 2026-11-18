#!/bin/bash
# Settles "run Fast_BCC per batch with NO filter, then at the end run
# BCC_Repair and give every 1-hop neighbour of a block node the same label."
#
# That end procedure is exactly BCC_FILTER_LAST=1 BCC_END_REPAIR=1.  It fails.
#
# NOTE: the A/B pair in end-repair-is-impossible.sh does NOT rule this out --
# both A and B come out correct under it, because the 1-hop step reads the
# committed block contents, not just the final H.  So it needs a direct test.
#
# 17 vertices, 16 edges, ONE EDGE PER BATCH (same graph as
# filter-schedule-counterexample.sh).  truth = 5 BCCs.
#   core 12-cycle : 0-1-11-12-13-4-8-7-10-9-2-3-0
#   pendants      : 0-6, 4-5, 9-16   plus the disjoint bridge 14-15
cd "$(dirname "$0")/.."
E="0 1  2 3  4 5  3 0  6 0  7 8  9 10  7 10  4 8  2 9  11 1  12 13  12 11  13 4  14 15  9 16"

echo "=== counts (truth 5) ==="
printf "  filter every round        : %s\n" "$(./build/case 16 17 $E 2>/dev/null | grep 'count  got')"
printf "  1-hop at end only         : %s\n" "$(BCC_FILTER_LAST=1 ./build/case 16 17 $E 2>/dev/null | grep 'count  got')"
printf "  1-hop at end + BCC_Repair : %s\n" "$(BCC_FILTER_LAST=1 BCC_END_REPAIR=1 ./build/case 16 17 $E 2>/dev/null | grep 'count  got')"

echo
echo "=== WHY: the block is demoted at round 14, two rounds before the end."
echo "    n_s going UP is vertices falling back out of a split block. ==="
printf "  filter every round : "; ./build/case 16 17 $E 2>/dev/null | grep 'batch @1[345]' | tr -s ' ' | tr '\n' ' '; echo
printf "  end-only filter    : "; BCC_FILTER_LAST=1 ./build/case 16 17 $E 2>/dev/null | grep 'batch @1[345]' | tr -s ' ' | tr '\n' ' '; echo

echo
echo "=== and so the blocks end up EMPTY: nothing for a 1-hop ring to rebuild ==="
echo "  filter every round:"; ./build/case 16 17 $E 2>/dev/null | grep -A3 'f\^-1 of each block' | tail -n +2
echo "  end-only filter:";    BCC_FILTER_LAST=1 ./build/case 16 17 $E 2>/dev/null | grep -A3 'f\^-1 of each block' | tail -n +2

echo
echo "=== repair then overshoots the other way: 6 -> 4, swallowing bridge {9,16} ==="
BCC_FILTER_LAST=1 BCC_END_REPAIR=1 ./build/case 16 17 $E 2>/dev/null | grep -A3 'bcc sets'

echo
echo "=== the flag you would key the 1-hop rule on is itself WRONG ==="
echo "  filter every round -- the block is flagged B and holds the cycle:"
BCC_SHOW=1 BCC_NAMES=1 ./build/case 16 17 $E 2>/dev/null | grep 'H   :' | tail -1
echo "  end-only -- the cycle sits in an OWN-node O2, and both blocks are empty:"
BCC_FILTER_LAST=1 BCC_SHOW=1 BCC_NAMES=1 ./build/case 16 17 $E 2>/dev/null | grep 'H   :' | tail -1
echo "  so '1-hop around every block node' paints around B0{} and B1{} -- which"
echo "  hold nothing -- and skips O2, the only node that actually holds a BCC."
echo "  Note also that vertex 9 is trapped INSIDE O2, though it is a cut vertex"
echo "  (16 hangs off it).  That is how the bridge {9,16} gets swallowed."

echo
echo "=== rate over 1400 random graphs ==="
echo "  filter every round        : splits 0   over-merges 0"
echo "  1-hop at end + BCC_Repair : splits 4   over-merges 22"
echo "  (reproduce: for s in 1 201 401 601 801 1001 1201; do \\"
echo "     BCC_FILTER_LAST=1 BCC_END_REPAIR=1 ./build/test_dense_pipeline \$s 200; done)"
