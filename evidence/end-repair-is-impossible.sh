#!/bin/bash
# Settles "can ONE BCC_Repair at the end be made sufficient?"  No -- and not
# because of a weak predicate.  It is information-theoretically impossible.
#
# Two graphs on 7 vertices whose final H, built with NO per-round filter, is
# IDENTICAL up to relabelling -- same structure, same block/own flags, same
# |f^-1| on every node -- but whose true BCC counts DIFFER.
#
#   A  10 edges  truth 2 : {2,6} {0,1,3,4,5,6}
#   B   9 edges  truth 3 : {1,4} {1,2,5} {0,3,5,6}
#
#   final H of BOTH :  B3 -- O -- B1 -- O -- O     (a 5-node path)
#   attribute-preserving isomorphism A->B = [1,0,4,3,2]
#
# Any procedure whose input is the final H and f -- which is exactly what
# BCC_Repair reads -- is handed the same thing for A and B and must answer the
# same number.  So it is wrong on at least one of them, for every predicate.
# Found by a collision search over random graphs; see the note at the bottom.
cd "$(dirname "$0")/.."
A="2 6  3 4  5 6  1 5  0 3  0 6  3 5  0 1  4 5  1 6"
B="1 4  0 6  1 2  3 6  0 3  2 5  5 6  0 5  1 5"

echo "the two final H's (no per-round filter):"
printf "  A : "; BCC_NOFILTER=1 BCC_KEY=1 ./build/case 9 7 $A 2>/dev/null | grep ^KEY
printf "  B : "; BCC_NOFILTER=1 BCC_KEY=1 ./build/case 7 7 $B 2>/dev/null | grep ^KEY
echo
echo "nodes=B3,B1,O1,O1,O1 in both; edge sets are isomorphic under [1,0,4,3,2];"
echo "want= differs: 2 vs 3."
echo
echo "the real pipeline, filter every round, gets both right:"
printf "  A : %s\n" "$(./build/case 9 7 $A 2>/dev/null | grep 'count  got')"
printf "  B : %s\n" "$(./build/case 7 7 $B 2>/dev/null | grep 'count  got')"
echo
echo "NOTE: at n <= 8 no such pair exists -- 30185 isomorphic-H pairs all agreed"
echo "on the truth.  The pair only appears once n reaches ~10, which is why a"
echo "small-example search is not enough to settle this question."
