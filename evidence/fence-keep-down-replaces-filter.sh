#!/bin/bash
# RESOLUTION of the line of inquiry in fence-keep-in-fastbcc.sh.
#
# Three earlier attempts to replace Compress-Graph's filter by refusing to
# fence edges inside Fast_BCC all failed (that script has a minimal witness
# for each).  A fourth works, and the reason the first three failed is the
# reason the fourth is the only possible shape.
#
#   BCC_FENCE_KEEP=new   keep this round's leaver links        28 splits/1400
#   BCC_FENCE_KEEP=skel  literal G' <- PLAINEDGES u Fc u D     == new
#   BCC_FENCE_KEEP=blk   keep EVERY tree edge at a block    10 over-merges/1400
#   BCC_FENCE_KEEP=down  keep only tree edges whose PARENT      0 / 0
#                        is a block
#
# ---------------------------------------------------------------------------
# WHY 'down' is the right predicate
#
# Un-fencing a tree edge does not merge two NODES.  Keep (c,p) with par(c)=p
# in G' and you get l[c] = l[p].  Read both through the LCH contract
# (include/extern_bcc/lch.cuh): l[x] is the class of x's PARENT edge.  So
#
#       l[c] = class of (c,p)        l[p] = class of (p, par(p))
#
# and the merge is { (c,p), (p,par(p)) } -- both edges incident to p.
# Un-fencing a tree edge is a merge AT THE PARENT ENDPOINT, pairing its
# up-edge with that one down-edge.
#
# What a block node B needs is every edge at B in one class, i.e. a merge AT
# B.  Therefore B must be the PARENT:
#
#   * every child c of B gets l[c] = l[B], so all of B's down-edges share one
#     class;
#   * B's own up-edge needs no union at all, because by the readout rule
#     beta(B,par(B)) = rep[deeper endpoint] = l[B] -- it is already that class;
#   * nothing merges at par(B), so no bridge above the block is absorbed.
#
# Mode 'blk' also un-fences B's up-edge (B as the CHILD).  That one merge
# lands at par(B) -- an own node, often a genuine cut vertex.  B's up-edge is
# the single edge at B whose un-fencing performs its merge somewhere else,
# and it is exactly what 'blk' over-merges with (W3: the bridge 3--5).
#
# Mode 'new' is orientation-dependent for the same reason: it is right only
# when the rooting happens to put the block above the leaver.  That is why it
# scores 28/1400 rather than failing uniformly, and why hand-built shapes
# would not reproduce the failures -- THE TRIGGER IS AN ORIENTATION, NOT A
# SHAPE.
#
#       root at 6:  6-4-0-B0-1'-5-7     par(1')=B0
#                   un-fence (1',B0) -> merges (B0,1') with (B0,0)   CORRECT
#       root at 7:  7-5-1'-B0-0-4-6     par(B0)=1'
#                   un-fence (B0,1') -> merges (B0,1') with (1',5)   WRONG
#                                       (1'--5 is a genuine bridge)
#
# Implemented as mode 4 in union_find_gpu_keepfence
# (src/fast_bcc/connected_components.cu): keep = (v < n_b_prev), where
# v = d_parent[idx].  One compare, same kernel, and it switches the filter
# off in gpu_compress.cu -- a REPLACEMENT, not an addition.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.."

W1="0 1  2 3  4 5  3 0  6 0  7 8  9 10  7 10  4 8  2 9  11 1  12 13  12 11  13 4  14 15  9 16"
W2="0 1  2 3  4 3  4 0  5 6  6 7  8 1  8 7  2 5  9 1  10 6  4 11"
W3="1 2  2 8  3 7  1 8  0 5  4 6  0 1  3 4  6 7  3 5"
W4="0 1  1 2  2 3  0 4  3 0  4 6  1 5  5 7"

row() { printf "  %-22s %s\n" "$1" "$(env $2 ./build/case $3 $4 $5 2>/dev/null | grep "count  got")"; }

echo "=== the four witnesses that killed the earlier modes ==="
for w in 1 2 3 4; do
  case $w in
    1) B=16; N=17; E="$W1"; T="17V/16E";; 2) B=12; N=12; E="$W2"; T="12V/12E";;
    3) B=4;  N=9;  E="$W3"; T="9V/10E  (kills blk AND new)";;
    4) B=4;  N=8;  E="$W4"; T="8V/8E   (kills 'permanent')";;
  esac
  echo "-- W$w  $T"
  row "DEFAULT (parent-is-blk)" ""                     $B $N "$E"
  row "v2: stock + filter"    "BCC_FENCE_KEEP=0"      $B $N "$E"
  row "fence-keep=blk"        "BCC_FENCE_KEEP=blk"    $B $N "$E"
  row "fence-keep=new"        "BCC_FENCE_KEEP=new"    $B $N "$E"
  row "no filter at all"      "BCC_NOFILTER=1"        $B $N "$E"
done

echo
echo "=== 'down' is the only mode that survives all four ==="
echo "  W3 is the decisive one: blk and new both give 4 where truth is 5, and"
echo "  no filter at all gives the right answer there by luck.  'down' is exact."

echo
echo "=== random regression: 60 seeds x 300 graphs = 17979 graphs ==="
echo "  (slow -- ~90 s serialised.  Set QUICK=1 for 8 x 200.)"
SEEDS=$( [ -n "$QUICK" ] && echo "1 164 500 1000 2000 3000 4242 9001" \
                         || seq 137 137 8220 )
N=$( [ -n "$QUICK" ] && echo 200 || echo 300 )
for mode in "" "0"; do
  sp=0; mg=0; g=0; bad=""
  for s in $SEEDS; do
    o=$(env ${mode:+BCC_FENCE_KEEP=$mode} ./build/test_dense_pipeline $s $N 2>/dev/null | tail -10)
    g=$((  g + $(echo "$o" | sed -n 's/.*graphs checked *: *//p') ))
    sp=$(( sp + $(echo "$o" | sed -n 's/.*real BCC split *: *//p') ))
    mg=$(( mg + $(echo "$o" | sed -n 's/.*over-merge *: *//p') ))
    echo "$o" | grep -q "all exact" || bad="$bad $s"
    echo "$o" | grep -qE "FOUND"    && bad="$bad ${s}!"
  done
  printf "  %-22s graphs %5d  splits %3d  over-merge %3d  %s\n" \
         "${mode:+v2-filter}${mode:-DEFAULT}" "$g" "$sp" "$mg" "${bad:+INEXACT:$bad}"
done
echo "  recorded: filter 17979/0/0 , down 17979/0/0 -- also no empty blocks,"
echo "  no false cut vertices, no thin blocks in either."

echo
echo "=== real graphs: identical block-cut structure, not just the same count ==="
D=/mnt/hdd/CS22S501/datasets/medium_datasets/ecl_graphs
for f in in-2004 as-Skitter coPapersDBLP eu-2005 kron_g500-logn20 delaunay_n24; do
  [ -f "$D/$f.egr" ] || continue
  for m in "" "0"; do
    o=$(env ${m:+BCC_FENCE_KEEP=$m} ./build/External_BCC -i $D/$f.egr -a ext-bcc -g -b 2000000 2>&1)
    printf "  %-18s %-9s bccs %-8s max-headed %s\n" "$f" "${m:+v2-filter}${m:-DEFAULT}" \
      "$(echo "$o" | sed -n 's/.*components in .* is: *//p')" \
      "$(echo "$o" | sed -n 's/.*max BCCs headed by one v: *//p')"
  done
done

echo
echo "=== and it is FASTER: the filter was a whole extra pass over H per round ==="
echo "  computeBCG, median of 5, serialised on a shared GPU:"
echo "    graph              bs          filter   down    gain"
echo "    in-2004            500000        101      92    8.9%"
echo "    in-2004            2000000        42      36   14.3%"
echo "    as-Skitter         500000         87      83    4.6%"
echo "    as-Skitter         2000000        36      33    8.3%"
echo "    coPapersDBLP       500000         72      69    4.2%"
echo "    coPapersDBLP       2000000        27      25    7.4%"
echo "    hollywood-2009     500000        299     284    5.0%"
echo "    hollywood-2009     2000000       112     101    9.8%"
echo "    eu-2005            500000         98      90    8.2%"
echo "    eu-2005            2000000        46      39   15.2%"
echo
echo "  The filter had to run after beta existed, so it was a separate phase:"
echo "  union the classes, then RE-DERIVE edge counts, anchors and the cut set"
echo "  from the merged labels.  'down' needs none of that -- the classes come"
echo "  out merged, so everything downstream is derived once."
