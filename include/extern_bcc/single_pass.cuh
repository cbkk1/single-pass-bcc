/******************************************************************************
* Single-pass (transformation-free) BCC: the de-merger.
*
* ID SCHEME. Supernode ids are split into two fixed regions:
*
*     [0, n)      original vertices, by index
*     [n, 2n)     blocks
*
* A standalone vertex is its own supernode, so its id is its own index; a
* vertex that has been absorbed points at a block id >= n. The two regions
* cannot collide, so
*
*     M[x] >= n   <=>   x currently sits inside a block
*
* and de-merging x is simply M[x] = x -- slot x belongs to vertex x and can
* never be taken by another supernode. That is why there is no fresh-id phase:
* the prefix-sum id minting in the paper's Phase 2 exists only to work around
* a dense id space, and is unnecessary here.
*
* #blocks <= n holds because the BCG only ever contracts: update_bcc_numbers
* maps every node to exactly one successor, so n_s never grows past |V|.
 ******************************************************************************/

#ifndef SINGLE_PASS_CUH
#define SINGLE_PASS_CUH

#include "extern_bcc/bcg_memory_utils.cuh"

// Exclusive prefix sum over n ints. The repo only ships an inclusive scan
// (fast_bcc.cu); the de-merger needs the exclusive form to turn per-edge
// output counts into write offsets.
//
// Uses the GPU_BCG::d_temp_storage member as a persistent CUB buffer rather
// than cudaMalloc/cudaFree per call, growing it on demand.
void excl_scan(const int* d_in, int* d_out, int n, GPU_BCG& g_bcg_ds);

// Running maximum. Used to hand every extraction pair the index at which its
// block's sorted run begins, which is what turns a flat sorted list into
// segments without a second sort.
void incl_max_scan(const int* d_in, int* d_out, int n, GPU_BCG& g_bcg_ds);

// Grows the shared CUB scratch buffer in place if it is smaller than `needed`.
void grow_temp_storage(GPU_BCG& g_bcg_ds, size_t needed);

// Phase 3 -- the de-merger.
//
// Every batch edge (u,v) whose endpoints lie in different supernodes is kept,
// and every endpoint sitting inside a block is pulled out of it via M[x] = x.
// What re-attaches the leavers to the block they came from is a RING, not a
// star: vertices leaving one block were pairwise 2-connected inside it, so
// hanging each off the block hub separately makes the hub a spurious
// articulation point and splits components that belong together. The ring over
// {remnant block} u {its leavers} preserves that 2-connectivity; one leaver
// degenerates to the single star edge, which is correct.
//
// The ring cannot be built per edge: which vertices leave a block is not known
// until the whole batch has been scanned. So the extraction pairs are collected,
// sorted by block, deduped, and only then rung -- which is also why an earlier
// per-edge "cycle" attempt measured worse than the star.
//
// Decisions are all taken against the PRE-batch mapping, matching the paper's
// parallel loop; the extractions are applied in a final separate launch.
//
// Appends to d_out starting at `base`, and returns the number of edges written.
long demerge(GPU_BCG& g_bcg_ds, long BatchSize, uint64_t* d_out, long base);

// One batch of the single-pass algorithm: phases 3, 4 and 5.
//
//   3. de-merge the batch onto the carried active edge set   (GPU)
//   4. biconnected components, and rebuild the block-cut graph from them
//                                                            (host, for now)
//   5. compose M[v] = I[M[v]]; upload the new BCG and filter  (GPU)
//
// Reads the batch from g_bcg_ds.d_edgelist and leaves the new active edge set
// in g_bcg_ds.updated_edgelist with g_bcg_ds.numEdges updated.
void single_pass_batch(GPU_BCG& g_bcg_ds, long BatchSize);

/*--------------------------------------------------------------------
 * The DENSE-ID path: Algorithm 3 end to end on the GPU, with Compress-Graph
 * (gpu_compress.cuh) standing in for the host Tarjan above.
 *
 * Node ids are renumbered densely every batch instead of living in fixed split
 * ranges. That makes every id-space-sized array shrink to |V(H_i)| -- measured
 * at 6-35% of n on real graphs -- at the price of an O(n) indirect rewrite of M
 * per batch and the return of the paper's Phases 1 and 2, which split ranges
 * make unnecessary.
 *
 * Carries n_s and n_b in GPU_BCG: dense ids have to be told where the block and
 * own-id ranges currently end, where split ranges know it statically.
 *-------------------------------------------------------------------*/

// Phases 1-3 for dense ids: mark, mint, emit. Appends to d_out from `base` and
// returns the number of edges written. Advances g.n_s to the transient n_t.
long demerge_dense(GPU_BCG& g, long BatchSize, uint64_t* d_out, long base);

// One batch, all on the GPU.
void single_pass_batch_gpu(GPU_BCG& g, long BatchSize);

#endif // SINGLE_PASS_CUH
