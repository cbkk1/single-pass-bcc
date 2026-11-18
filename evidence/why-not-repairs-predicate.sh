#!/bin/bash
# "Filter and BCC_Repair are the same operation -- so why the filter?"
# They ARE the same union. They differ in the PREDICATE, and the predicate is
# the whole point:
#   BCC_Repair : |f^-1(v)| == 1   -- is v a real VERTEX?   (a counting test)
#   filter     : v < n_b_prev     -- is v already a BLOCK? (a structural test)
# Mid-stream these differ, because a legitimate block can be de-merged down to
# exactly ONE resident.  Counting then calls it a real vertex and skips it.
#
# 9 vertices, 8 edges, one edge per batch.
#   cycle 1-3-2-4-1 -> BCC {1,2,3,4};  bridges 0-1, 2-5, 3-6, 7-8;  truth = 5
# At @6 the cycle closes.  1, 2 and 3 are all cut vertices (pendants 0, 5, 6),
# so the block keeps a SINGLE resident: B0{4}.  At @7 three beta classes touch
# B0; the filter must merge them.  BCC_Repair's predicate will not, because
# |f^-1(B0)| = |{4}| = 1.
cd "$(dirname "$0")/.."
E="0 1  2 3  2 4  2 5  1 3  6 3  4 1  7 8"

for p in id dis cnt; do
  printf "pred=%-4s : %s\n" $p "$(BCC_PRED=$p ./build/case 8 9 $E 2>/dev/null | grep 'count  got')"
done
echo
echo "--- what pred=cnt produces ---"
BCC_PRED=cnt ./build/case 8 9 $E 2>/dev/null | tail -3
echo
echo "--- the deciding round (@7), beta class by beta class ---"
echo "  pred=cnt:"
BCC_PRED=cnt BCC_BETA=1 ./build/case 8 9 $E 2>/dev/null | awk '/n_b_prev=1/,/after batch @7/' | grep beta
echo "  pred=id:"
BCC_PRED=id  BCC_BETA=1 ./build/case 8 9 $E 2>/dev/null | awk '/n_b_prev=1/,/after batch @7/' | grep beta
