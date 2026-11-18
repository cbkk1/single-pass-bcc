#!/bin/bash
# Tests the proposal "don't repair the split afterwards -- just stop Fast_BCC
# from fencing the leaver's link in the first place."
#
# The diagnosis behind it is CORRECT, and the de-merger says so itself
# (single_pass.cu, dense_scatter_kernel):
#
#   "ONE link per leaver: a block is a single node, and the filter in
#    Compress-Graph is what re-establishes 2-connectivity through it."
#
# A leaver gets ONE edge back to the block it left, so f_v <= low_u and
# l_v >= high_u both hold and the fence test reads it as a bridge -- cutting
# the leaver off from a remnant it is genuinely 2-connected to.
#
# Implemented as union_find_gpu_keepfence (fast_bcc/connected_components.cu),
# selected by BCC_FENCE_KEEP, which also switches OFF Compress-Graph's filter
# so this is a REPLACEMENT, not an addition:
#
#   BCC_FENCE_KEEP=new   exempt only THIS round's leaver links from the fence
#                        test (one end < n_b_prev, other end >= n_s_prev)
#   BCC_FENCE_KEEP=skel  the literal G' <- PLAINEDGES(T) u Fc u D: stock CC,
#                        then union every leaver link unconditionally
#   BCC_FENCE_KEEP=blk   exempt every tree edge at a carried block node
#
# RESULT: neither is equivalent to the filter. They miss in OPPOSITE directions.
cd "$(dirname "$0")/.."

W1="0 1  2 3  4 5  3 0  6 0  7 8  9 10  7 10  4 8  2 9  11 1  12 13  12 11  13 4  14 15  9 16"
W2="0 1  2 3  4 3  4 0  5 6  6 7  8 1  8 7  2 5  9 1  10 6  4 11"
# MINIMAL witness: 9 verts, 10 edges, bs=4.  Shrunk by greedy edge+pair
# deletion with a two-sided oracle, from test_dense_pipeline seed 164.
#   triangle {1,2,8}   4-cycle {3,4,6,7}   bridges 0-1, 0-5, 3-5
#   truth 5 : {0,1} {0,5} {1,2,8} {3,4,6,7} {3,5}
# It kills BOTH modes at once.  Note that hand-built shapes (cycle + two
# bridges, cycle + bridge into a hub, two cycles sharing a cut vertex) do NOT
# reproduce this at any batch size: the trigger needs a particular
# spanning-tree orientation, which is why it is only 10 cases per 1400.
W3="1 2  2 8  3 7  1 8  0 5  4 6  0 1  3 4  6 7  3 5"

row() { printf "  %-22s %s\n" "$1" "$(env $2 ./build/case $3 $4 $5 2>/dev/null | grep "count  got")"; }

echo "=== W1: 17V/16E, one edge per batch (truth 5) ==="
row "v2: stock + filter" "BCC_FENCE_KEEP=0" 16 17 "$W1"
row "fence-keep=blk" "BCC_FENCE_KEEP=blk" 16 17 "$W1"
row "fence-keep=new" "BCC_FENCE_KEEP=new" 16 17 "$W1"
row "fence-keep=skel" "BCC_FENCE_KEEP=skel" 16 17 "$W1"
row "no filter at all" "BCC_NOFILTER=1" 16 17 "$W1"
echo "  -> 'new' is no better than no filter: the block that breaks here is torn"
echo "     apart by attachments carried in from EARLIER rounds, not by this"
echo "     round's leaver links, so exempting only the new links changes nothing."

echo
echo "=== W2: 12V/12E, one edge per batch (truth 4) ==="
row "v2: stock + filter" "BCC_FENCE_KEEP=0" 12 12 "$W2"
row "fence-keep=blk" "BCC_FENCE_KEEP=blk" 12 12 "$W2"
row "fence-keep=new" "BCC_FENCE_KEEP=new" 12 12 "$W2"

echo
echo "=== W3: MINIMAL -- 9V/10E, bs=4 (truth 5).  Breaks BOTH modes ==="
row "v2: stock + filter" "BCC_FENCE_KEEP=0"                   4 9 "$W3"
row "fence-keep=blk" "BCC_FENCE_KEEP=blk"   4 9 "$W3"
row "fence-keep=new" "BCC_FENCE_KEEP=new"   4 9 "$W3"
row "fence-keep=skel" "BCC_FENCE_KEEP=skel" 4 9 "$W3"
echo
echo "  the last round, in original vertex names:"
printf "    filter : "; BCC_FENCE_KEEP=0 BCC_SHOW=1 BCC_NAMES=1 ./build/case 4 9 $W3 2>/dev/null | grep 'H   :' | tail -1 | sed 's/^ *H *: *//'
printf "    blk    : "; BCC_FENCE_KEEP=blk BCC_SHOW=1 BCC_NAMES=1 ./build/case 4 9 $W3 2>/dev/null | grep 'H   :' | tail -1 | sed 's/^ *H *: *//'
echo
echo "  Vertex 3 is a genuine cut vertex -- the bridge 3-5 hangs off it -- so the"
echo "  filter keeps it OUTSIDE the block, as B1{4,6,7}--3, and 5--3 stays a"
echo "  bridge.  Under blk, 3 is ABSORBED (B1{3,4,6,7}) and the bridge is"
echo "  re-pointed at the block as B1--5, so {3,5} folds into {3,4,6,7}:"
BCC_FENCE_KEEP=0 ./build/case 4 9 $W3 2>/dev/null | grep -A2 'bcc sets' | sed 's/^/    /'
BCC_FENCE_KEEP=blk ./build/case 4 9 $W3 2>/dev/null | grep -A3 'bcc sets' | sed 's/^/    /'

echo
echo "=== 'keep the FRESHLY ADDED de-merge edges' -- what it actually covers ==="
echo "  G' <- PLAINEDGES(T) u Fc u D, with D = this round's leaver links, is"
echo "  mode 'skel'.  It matches 'new' everywhere, because a de-merge link can"
echo "  only be LOST if the spanning tree made it a tree edge and the fence test"
echo "  then dropped it: as a BACK edge it already forces its tree path"
echo "  non-fence, and as a CROSS edge it is covered by the spanning forest Fc."
echo
echo "  The reason it is not enough is that the edges needing a union at a block"
echo "  node are mostly NOT fresh de-merge links.  BCC_BLKEDGE=1 classifies them:"
BCC_FENCE_KEEP=0 BCC_BLKEDGE=1 ./build/case 16 17 $W1 2>/dev/null | grep -E 'after batch @1[345]|blkedge' | sed 's/^/    /'
echo
echo "  Round 14 is exactly where the filter must fire (see"
echo "  filter-schedule-counterexample.sh: deferring past 14 loses a BCC) and it"
echo "  has ZERO fresh de-merge links.  There is nothing for the rule to keep."
echo
echo "  Two reasons a block-incident edge is not a fresh de-merge link:"
echo "   1. an OLD leaver's link.  A leaver that is a cut vertex is never"
echo "      re-absorbed, so its link persists for the rest of the stream -- but"
echo "      L[u] >= n_s_prev holds only in the round it was minted, so it counts"
echo "      as 'fresh' exactly once and is carried forever after."
echo "   2. a never-absorbed member.  own_kernel keeps a cut vertex of the"
echo "      compressed class OUT of the block, and the class edge joining it is"
echo "      rebuilt as block--vertex.  That edge was never a de-merge edge at all."
echo "  Measured over 400 random graphs (442 compress rounds with a carried block):"
echo "    fresh de-merge links  4221  (54.7%)"
echo "    carried edges         3490  (45.3%)"
echo "    rounds with ZERO fresh links but >=1 carried edge:  30/442 (6.8%)"
echo
echo "  So the right predicate is not 'fresh de-merge edge' but 'ANY edge at a"
echo "  block node' -- which is exactly the filter's x < n_b_prev."

echo
echo "=== and WHY even that cannot move into the skeleton (mode blk) ==="
echo "  beta is defined as rep[deeper endpoint of the edge] (beta_kernel)."
echo "  The fence test feeds rep[], so declining to fence an edge does not"
echo "  relabel THAT edge alone -- it moves rep[] for a whole component, and"
echo "  every other edge whose deeper endpoint sits in that component silently"
echo "  changes class too.  In W3 that swept the bridge 3--5 into the block's"
echo "  class, so node 3 stopped reading as a cut vertex and was absorbed."
echo "  The filter runs AFTER beta exists and unions beta LABELS, so it can"
echo "  merge exactly the classes at a block node and nothing else."

echo
echo "=== rate over 1400 random graphs (7 seeds x 200) ==="
echo "  filter (current)   splits  0   over-merges  0    <- clean"
echo "  fence-keep=blk     splits  0   over-merges 10    <- too broad"
echo "  fence-keep=new     splits 28   over-merges  0    <- too narrow"
echo "  fence-keep=skel    same as new (literal G'uD)"
echo "  (reproduce: for s in 1 201 401 601 801 1001 1201; do \\"
echo "     BCC_FENCE_KEEP=blk ./build/test_dense_pipeline \$s 200; done)"

echo
echo "=== and why marking the de-merge links PERMANENTLY does not fix it ==="
# cycle 0-1-2-3-0 with pendant 0-4 arriving BEFORE the cycle closes, so vertex 0
# is already a cut vertex when the block forms and is EXCLUDED at compression
# time.  Its edge to the block is a rebuilt cycle edge -- it was never a
# de-merge link, because 0 was never inside a block to be pulled out of.
W4="0 1  1 2  2 3  0 4  3 0  4 6  1 5  5 7"
echo "  truth 5 : {0,1,2,3} {0,4} {4,6} {1,5} {5,7}"
BCC_FENCE_KEEP=0 BCC_SHOW=1 BCC_NAMES=1 BCC_BLKEDGE=1 ./build/case 4 8 $W4 2>/dev/null \
    | grep -E 'after batch|H   :|blkedge' | sed 's/^/    /'
echo
row "v2: stock + filter" "BCC_FENCE_KEEP=0"                  4 8 "$W4"
row "fence-keep=new" "BCC_FENCE_KEEP=new"  4 8 "$W4"
row "fence-keep=blk" "BCC_FENCE_KEEP=blk"  4 8 "$W4"
echo
echo "  At the last round the block has TWO edges: (B,0) the cycle remnant and"
echo "  (B,1) a de-merge link.  The only de-merge link there IS the fresh one, so"
echo "  'keep every de-merge link permanently' and 'keep only this round's' do"
echo "  exactly the same thing here -- neither keeps (B,0) -- and both give 6."
echo
echo "  A leaver that stayed out and a never-absorbed member are the SAME object:"
echo "  a vertex of the block's BCC that is a cut vertex and so sits outside the"
echo "  block node.  Only the history differs -- excluded when the block formed,"
echo "  or pulled out later.  H does not distinguish them and both need the same"
echo "  treatment, so a rule phrased over de-merge links can never be complete."
echo "  Completing it means marking the block's neighbours, which is exactly the"
echo "  filter's x < n_b_prev -- one integer compare, and no history to carry."

echo
echo "=== RESOLVED: there IS a fourth mode, and it works ==="
echo "  BCC_FENCE_KEEP=down -- keep only tree edges whose PARENT is a block."
echo "  Un-fencing (c,p) merges the two edges at p, not the two nodes, so the"
echo "  block must be the parent.  See evidence/fence-keep-down-replaces-filter.sh:"
echo "  exact on all four witnesses here, 0/0 over 17979 random graphs, identical"
echo "  block-cut structure on 6 real graphs, and 4-15% faster than the filter."
