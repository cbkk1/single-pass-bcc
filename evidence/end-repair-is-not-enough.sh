#!/bin/bash
# "Is BCC_Repair at the end sufficient for the stream?"  No -- and on this
# graph it is not merely insufficient, it is INERT.
#
# 9 vertices, 8 edges, one edge per batch:
#   cycle 1-3-2-4-1 -> BCC {1,2,3,4};  bridges 0-1, 2-5, 3-6, 7-8;  truth = 5
#
# At @6 the cycle closes.  1, 2 and 3 are all cut vertices (pendants 0, 5, 6),
# so the block keeps a SINGLE resident:  B0{4}--1  B0{4}--2  B0{4}--3
# At @7, with no filter, the three beta classes at B0 each become their own
# block, so the one real BCC fragments into three:
#   B0{}--4 B0{}--1  B1{}--4 B1{}--2  B2{}--4 B2{}--3
#
# Now run BCC_Repair on that final H.  Its entire test is |f^-1(v)| == 1, and
# in this H EVERY node has exactly one preimage vertex:
#   3:{4} 4:{0} 5:{1} 6:{2} 7:{3} 8:{5} 9:{6} 10:{7} 11:{8}
# So repair finds no false cut vertex and changes nothing.  The error is in the
# STRUCTURE (three blocks where there should be one), not in any preimage
# count -- and preimage counts are all repair can see.
cd "$(dirname "$0")/.."
E="0 1  2 3  2 4  2 5  1 3  6 3  4 1  7 8"

printf "  %-26s %s\n" "filter every round" \
  "$(./build/case 8 9 $E 2>/dev/null | grep 'count  got')"
printf "  %-26s %s\n" "no filter, no repair" \
  "$(BCC_NOFILTER=1 ./build/case 8 9 $E 2>/dev/null | grep 'count  got')"
printf "  %-26s %s\n" "no filter + END REPAIR" \
  "$(BCC_NOFILTER=1 BCC_END_REPAIR=1 ./build/case 8 9 $E 2>/dev/null | grep 'count  got')"
echo
echo "  final H without the filter, and every node's |f^-1|:"
BCC_NOFILTER=1 BCC_NAMES=1 BCC_SHOW=1 ./build/case 8 9 $E 2>/dev/null \
  | grep -E "^        H   :" | tail -1
BCC_NOFILTER=1 BCC_SHOW=1 ./build/case 8 9 $E 2>/dev/null \
  | sed -n '/so |f\^-1(x)| per own-node id/,+1p'
echo
echo "  what no filter + END REPAIR returns:"
BCC_NOFILTER=1 BCC_END_REPAIR=1 ./build/case 8 9 $E 2>/dev/null | tail -2
