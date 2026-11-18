#ifndef FAST_BCC_CUH
#define FAST_BCC_CUH

#include "extern_bcc/bcg_memory_utils.cuh"

void Fast_BCC(GPU_BCG& g_bcg_ds, const bool isLastBatch);

// Steps 1-3 of Fast_BCC only -- spanning tree + Euler tour, the low/high tags,
// and the connectivity pass that assigns implicit-BCC labels. On return:
//
//   g_bcg_ds.d_rep[v]       implicit-BCC label of v: the block v hangs BELOW,
//                           i.e. the block holding v's parent edge. Every node
//                           of a block except its topmost one carries the
//                           block's label; the topmost one carries the label of
//                           the block above it.
//   g_bcg_ds.d_parent[v]    v's parent in the rooted spanning tree (v if root)
//   g_bcg_ds.d_first_occ[v] v's first occurrence in the Euler tour
//
// Those three give a per-EDGE block label, which is what Compress-Graph needs
// and what d_rep alone cannot be: a cut vertex belongs to several blocks, so no
// per-vertex labelling of them exists. See gpu_compress.cu.
//
// Deliberately stops short of finalise_labels() and update_bcc_numbers(): the
// first mutates d_rep[root], the second overwrites d_rep with the dense
// [0, n_blocks) renumbering. Compress-Graph does its own renumbering, in
// aligned pairs, and derives cut status from the per-edge labels.
//
// FOREST-CAPABLE. cuda_euler_tour takes the component count c and tours each
// tree in its own slice of a shared index space, so a disconnected input needs
// no augmentation. The caller must fill the per-component metadata first --
// see forest_metadata() in gpu_compress.cu.
void Fast_BCC_labels(GPU_BCG& g_bcg_ds);

#endif // FAST_BCC_CUH