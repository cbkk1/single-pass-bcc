/******************************************************************************
* Single-pass BCC, phase 4 -- host-side biconnected components.
*
* A temporary stand-in for GPU Fast-BCC. Fast-BCC assumes the active edge set
* is a single connected component (hardcoded root, E = 2N-2, roots_count = 1);
* dropping the pass-1 spanning tree makes it a forest, so it cannot be used
* until euler.cu grows multi-root support. This runs on the host, where forests
* cost nothing, and validates the rest of the pipeline first.
*
* Iterative, not recursive: one DFS frame per stack entry. The recursive form
* in the serial prototype overflows the stack past a few hundred thousand
* vertices.
 ******************************************************************************/

#ifndef SERIAL_BCC_CUH
#define SERIAL_BCC_CUH

#include <cstdint>
#include <vector>

// Biconnected components over a SPARSE node id space.
//
//   edges      packed (lo << 32) | hi, every id < num_nodes
//   n          original vertex count; blocks are numbered upward from n
//   num_nodes  size of the id space (V_max)
//
// On return, for every node id s:
//
//   I[s] = n + rank   s is non-cut and its BCC was compressed  (s is absorbed)
//   I[s] = s          s is a cut vertex, s is isolated, or its BCC is a
//                     single edge -- bridges are deliberately not compressed,
//                     matching the serial prototype
//
// `live` marks node ids that are currently supernodes, i.e. that hold at least
// one original vertex. It is needed because a block can SURVIVE a round -- if
// it turns out to be a cut vertex it keeps its id and its residents -- and a
// newly formed block must not be handed that same id. A block with no incident
// edges is invisible in `edges`, so this cannot be derived from the edge list.
//
// iscut[s] flags articulation points. Returns the number of blocks allocated.
// New and surviving blocks together stay within [n, 2n): blocks are edge
// disjoint and each spans at least 3 vertices, so there are at most (n-1)/2.
//
// out_edges, when given, receives the block-cut graph for the NEXT round,
// rebuilt from scratch rather than by relabelling these edges in place:
//
//   a bridge            -> kept, both endpoints mapped through I
//   a compressed block  -> one edge to each of its cut vertices
//   anything internal   -> dropped
//
// Relabelling cannot express the last line: an edge between two cut vertices of
// the same block maps to itself and survives looking exactly like a bridge.
// The result is a forest, so it holds fewer than num_nodes edges. It may repeat
// an edge across components, so it still wants a dedup pass.
int serial_bcc(const std::vector<uint64_t>& edges,
               int n, int num_nodes,
               const std::vector<char>& live,
               std::vector<int>& I, std::vector<char>& iscut,
               std::vector<uint64_t>* out_edges = nullptr);

// How many bridges a FINISHED block-cut graph holds -- the edges of it that are
// biconnected components in their own right, as opposed to the structural links
// and rings that hold blocks to their cut vertices.
//
// A bridge is a component of exactly one edge that touches no block id. Both
// halves matter: "one edge" alone would also count a ring edge running between
// two cut vertices of one block, and "no block endpoint" alone would also count
// the link from a block to its single cut vertex, where the ring degenerates.
//
//   num_nodes  size of the id space (V_max); ids >= n are blocks
long bcg_bridge_count(const uint64_t* edges, long m, int n, int num_nodes);

// Ground truth: the exact number of biconnected components of the WHOLE graph,
// by one serial Tarjan pass over the entire edge stream. Shares its DFS with
// serial_bcc above.
//
// This is the reference the streaming pipeline is measured against, not part of
// it -- it reads every edge at once, which is the cost the single-pass
// algorithm exists to avoid. Use it to check a run, not to produce one.
//
//   edges  packed (lo << 32) | hi, every id < n; not copied
//   m      number of packed edges
//   n      vertex count
//
// A single-edge component is a bridge and is reported separately via
// out_bridges. Isolated vertices are NOT counted as components -- whether a
// lone vertex is a trivial BCC is a convention, so the count is returned
// separately via out_isolated and the caller can add it if they want. Both
// out-params may be NULL.
long serial_bcc_count(const uint64_t* edges, long m, int n,
                      long* out_bridges = nullptr, long* out_isolated = nullptr);

// Reference edge->block partition, for differential testing of the GPU
// Compress-Graph. One serial Tarjan pass over `edges`, returning
//
//   bcc_of_edge[e]  the block id of edge e, in [0, #blocks)
//   iscut[s]        articulation-point flag per node
//   bcc_edges[j]    how many edges block j holds (1 == a bridge)
//
// and the number of blocks. Block ids are arbitrary, so compare PARTITIONS,
// never ids.
int serial_bcc_partition(const uint64_t* edges, long m, int num_nodes,
                         std::vector<int>& bcc_of_edge,
                         std::vector<char>& iscut,
                         std::vector<long>& bcc_edges);

#endif // SERIAL_BCC_CUH
