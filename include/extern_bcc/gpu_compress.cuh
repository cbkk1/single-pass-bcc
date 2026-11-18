/******************************************************************************
* Compress-Graph on the GPU, using Abhijeet's Fast_BCC as the BCC subroutine.
*
* This is Algorithm 4 of the single-pass write-up: given the transient graph
* that the de-merger produced, compute its biconnected components, renumber the
* nodes densely, and REBUILD the block-cut graph for the next batch.
*
* THREE THINGS FAST_BCC DOES NOT GIVE US, AND HOW EACH IS OBTAINED
*
* 1. Per-edge block labels. Fast_BCC's d_rep is one label per NODE, and a cut
*    vertex belongs to several blocks at once, so no per-node labelling of them
*    can exist. What d_rep actually holds is "the block this node hangs below",
*    i.e. the block containing the node's parent edge. Every node of a block
*    except its topmost one therefore carries that block's label. So for an
*    edge, the label is read off the endpoint that is NOT the block's top -- and
*    that is always the endpoint deeper in the Euler tour:
*
*        beta(x,y) = d_rep[ the endpoint with the larger d_first_occ ]
*
*    For a tree edge the deeper endpoint is the child, whose parent edge IS this
*    edge. For a back edge (y inside x's subtree) x may be the block's top, and
*    y is deeper. For a cross edge between two subtrees the block's top is their
*    meeting point, which is neither endpoint, so both labels agree and either
*    will do. One rule covers all three.
*
* 2. A cut-vertex set we can trust across the augmentation below. Rather than
*    Fast_BCC's iscutVertex, cut status is derived straight from the definition:
*
*        cut[x]  <=>  |{ beta(e) : e incident to x }| >= 2
*
*    which is id-free, needs no root special case, and is automatically blind to
*    the edges the augmentation adds, because those are simply not scanned.
*
* 3. Forest support. cuda_euler_tour hardcodes E = 2N-2 and create_dup_edges
*    compacts around exactly one root, so a disconnected input reads garbage.
*    Dropping the pass-1 spanning tree makes the active edge set a forest, so
*    this has to be handled. Instead of teaching the Euler tour about multiple
*    roots (~5 kernels), we add a VIRTUAL SUPER-ROOT: one extra node joined by
*    one edge to one representative of every component. The result is connected,
*    and Fast_BCC is used completely unmodified.
*
*    That is sound because the added structure is a pendant tree: every added
*    edge is a bridge, so it forms its own singleton block and cannot merge,
*    split, or otherwise disturb any block of the original graph. It does make
*    each attachment point look like a cut vertex, but that never reaches us --
*    cut status is recomputed by (2) over the original edges only, so the
*    augmentation's own edges are invisible to it.
*
* ID SCHEME PRODUCED. Dense, in two ranges:
*
*        [0, 2*n_b)      blocks, in ALIGNED PAIRS (p, p+1) with p even
*        [2*n_b, n_s)    everything that keeps its own identity
*
* so "s is a block" is the single test s < 2*n_b, and "s is a primary half" is
* s even. A block is two nodes joined by an edge, and every attachment point
* joins BOTH halves, because one node cannot express a 2-connection: a lone
* attachment edge is a bridge, which makes the block a false articulation point
* and strands its remaining residents.
 ******************************************************************************/

#ifndef GPU_COMPRESS_CUH
#define GPU_COMPRESS_CUH

#include "extern_bcc/bcg_memory_utils.cuh"

// What Compress-Graph returns to the caller.
struct CompressResult {
    int n_s;    // node count of the rebuilt graph  (= 2*n_b + n_o)
    int n_b;    // blocks in it; they own [0, 2*n_b) in aligned pairs
    long m_s;   // edges in the rebuilt graph
};

// Compress-Graph. Reads `n_t` nodes and `m_a` edges from g.updated_edgelist
// (Fast_BCC hardcodes that member as its input) and leaves the rebuilt graph in
// g.d_edge_buffer, with I[] in g.d_newId.
//
// The caller composes the mapping itself -- M[v] = I[M[v]] -- and is expected
// to filter the result: the rebuild emits the same attachment edge once per
// internal edge of a block, so duplicates are normal and deliberate.
CompressResult compress_graph_gpu(GPU_BCG& g, int n_t, long m_a);

// Per-edge block labels and cut status alone, without the renumbering or the
// rebuild. Exposed for differential testing against the serial Tarjan; the
// pipeline itself has no reason to call this.
//
// Writes n_t entries to g.d_first_beta / g.d_own and m_a to g.d_beta, and
// returns through g.iscutVertex. Block labels are arbitrary node ids, so
// compare PARTITIONS, never ids.
void bcc_labels_gpu(GPU_BCG& g, int n_t, long m_a);

// Debug hook: with BCC_CHECK=1 set, compares this batch's block partition and
// cut set against the serial Tarjan on the same transient graph. No-op otherwise.
void compress_check(GPU_BCG& g, int n_t, long m_a, int batch);

#endif // GPU_COMPRESS_CUH
