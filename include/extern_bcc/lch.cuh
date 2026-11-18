/******************************************************************************
* LCH readout: label[] + componentHead[], the output format of Algorithm 1.
*
* A cut vertex belongs to several biconnected components at once, so no
* per-vertex labelling can name them all. LCH sidesteps that by giving each
* component ONE label class plus ONE head:
*
*     BCC(j)  =  { v : label[v] = j }  u  { componentHead[j] }
*
* Why one head is enough: root the block-cut forest. Every component has
* exactly one vertex above it and any number below. Label each vertex by the
* component holding its PARENT edge, and a component's class picks up all its
* lower vertices; only its top one is missing, and that is the head. A vertex
* cutting k components is therefore the head of k-1 of them and labelled by the
* k-th -- so multi-membership works at any multiplicity.
*
* Computed on the host, once, after the stream ends. The caller already copies
* M and the final edge set back for the count, so this adds no data movement,
* and it is a single O(n + n_s) pass against O(k*n) for the streaming itself.
 ******************************************************************************/

#ifndef LCH_CUH
#define LCH_CUH

#include <cstdint>
#include <vector>

// Reads the FINISHED block-cut graph and produces the LCH form.
//
//   M        M[v] for every original vertex, v in [0, n)
//   edges    the final active edge set, packed (lo << 32) | hi
//   n_s      node count of the final BCG
//   n_b      blocks in it; they own [0, 2*n_b) in aligned pairs,
//            everything that kept its own identity owns [2*n_b, n_s)
//
// On return:
//
//   label[v]           the component v is labelled by, or -1 if v is the top of
//                      its tree (it is then only ever a head) or is isolated
//   head[j]            the component head of label j, or -1 when component j is
//                      a whole connected component and so has no vertex above it
//
// Returns the number of distinct components. Block components are numbered
// [0, n_b) by their pair; bridges follow them.
int compute_lch(const std::vector<int>& M, int n,
                const std::vector<uint64_t>& edges, int n_s, int n_b,
                std::vector<int>& label, std::vector<int>& head);

#endif // LCH_CUH
