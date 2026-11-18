/******************************************************************************
* Functionality: Memory Management
* Handles allocation, deallocation, and initialization of all variables
 ******************************************************************************/

#ifndef BCC_MEM_UTILS_H
#define BCC_MEM_UTILS_H

#include <cuda_runtime.h>
#include "utility/cuda_utility.cuh"

class GPU_BCG {
public:
    GPU_BCG(int, long, long);
    // ~GPU_BCG();

    void init();

    // These two denotes the #f vertices and #f edges in the pbcg graph H_i
    // ____________________________________________________________________ //
    int numVert;
    long numEdges;
    // -------------------------------------------------------------------- //

    int last_root = -1;

    long batchSize;
    long E;  // Double the edge count (e,g. for edge (2,3), (3,2) is also counted)
    long max_allot;

    // Allocation size of updated_edgelist / d_edge_buffer / d_flags. Distinct
    // from numEdges, which is a LIVE counter the batch loop rewrites -- bounds
    // checks must use this one.
    long edge_capacity;

    const int orig_numVert;
    const long orig_numEdges;

    // Size of the supernode id space. Vertices own [0, numVert); blocks own
    // [numVert, V_max) in ALIGNED PAIRS, so an id is a block's primary half
    // exactly when (id - numVert) is even. Residents always map to a primary.
    int V_max;

    // Ceiling on the node count the BCC step ever sees, i.e. on the transient
    // n_t of Algorithm 3. Sized for the DENSE id scheme:
    //
    //     n_s <= 2 * numVert        blocks take an aligned pair, and there are
    //                              at most (n-1)/2 of them
    //     + 2 * batchSize          the de-merge mint: a batch of b edges
    //                              extracts at most 2b residents
    //     + 1                      the virtual super-root that makes a forest
    //                              connected for Fast_BCC
    //     + 1                      slack
    //
    // Every Fast_BCC array is sized on this rather than on numVert, because the
    // graph handed to it is the transient one, not the original.
    int V_fb;

    /* Pointers for dynamic memory */
    // a. Input edge stream (host pinned memory)
    const uint64_t* h_edgelist;
    
    // b. device arrays
    // 1. Pointers for batch of edges
    uint64_t *d_edgelist;

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ 2. Batch Spanning Tree data-structures_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
    uint64_t* d_parentEdge;
    int* d_componentParent;
    int* d_rep;
    
    // pinned memory for flags
    int* h_flag;
    int* h_shortcutFlag;
    
    // respective device flags
    int* d_flag;
    int* d_shortcutFlag;
    // _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- //

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ 3. BCC data-structures_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- //
    uint64_t* updated_edgelist;   // This list contains all edges after removing self-loops and duplicate edges
    uint64_t* d_edge_buffer;      // This is an intermediate buffer where we store the previous set of edges plus new batch of edges

    int* d_mapping = NULL;
    int* d_imp_bcc_num = NULL;
    long* h_num_selected_out = NULL;
    long* d_num_selected_out = NULL;
    void *d_temp_storage =  NULL;
    size_t temp_storage_bytes = 0;   // capacity of d_temp_storage, grown on demand
    unsigned char *d_flags = NULL;

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ b. Eulerian Tour data-structures_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- //
    int *d_edges_to;
    int *d_edges_from;

    uint64_t *d_index;
    int *d_next;
    int *d_roots;

    int *d_first;
    int *d_last;

    int *succ;
    int *devRank;

    int *notAllDone;
    ull *devRankNext;
    int *devNotAllDone;

    int *d_parent;
    int *d_level;
    int *devW1Sum;
    int *d_first_occ;
    int *d_last_occ;
    int *iscutVertex;
    // _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ c. Assigning Tag starts -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- //
    int* d_w1;
    int* d_w2;
    int* d_a1;
    int* d_a2;
    int* d_na1;
    int* d_na2;
    int* d_left;
    int* d_right;
    int* d_low;
    int* d_high;
    int* d_fg;

    // making the bcc labels continous
    int* d_bcc_flag;
    int* h_max_ps_bcc;
    int* h_max_ps_cut_vertex;

    // repair
    int* d_counter;
    int* d_isFakeCutVertex;

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ Single-pass de-merger _-_-_-_-_-_-_-_-_-_-_-_-_-_-_- //
    int* d_mark = NULL;             // vertices to pull out of a block, by original vertex
    int* d_edge_out_count = NULL;   // per-batch-edge output count (0..5)
    int* d_edge_out_offset = NULL;  // exclusive scan of the above -> write offsets

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ GPU Compress-Graph _-_-_-_-_-_-_-_-_-_-_-_-_- //
    // All node-indexed arrays here are sized V_fb, since Compress-Graph runs on
    // the transient graph. See gpu_compress.cu for what each holds.
    int* d_cc = NULL;            // union-find labels, for finding the components
                                 // to hang off the super-root
    int* d_beta = NULL;          // per-edge block label (numEdges)
    int* d_blk_edges = NULL;     // edges per block
    int* d_blk_flag = NULL;      // 1 iff that block compresses. Sized V_fb + 1:
                                 // compress_graph_gpu reads index n_t to total the scan.
    int* d_blk_struct = NULL;    // 1 iff it holds a carried block's pair edge
    int* d_first_beta = NULL;    // first block seen at a node, -1 if isolated
    int* d_own = NULL;           // 1 iff the node keeps its own identity
    int* d_scan_b = NULL;        // exclusive scan of d_blk_flag  (V_fb + 1)
    int* d_scan_o = NULL;        // exclusive scan of d_own       (V_fb + 1)
    int* d_newId = NULL;         // I[]: old node -> new node
    int* d_cnt = NULL;           // per-edge emission count       (numEdges + 1)
    int* d_off = NULL;           // exclusive scan of the above   (numEdges + 1)

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ Dense-id de-merger _-_-_-_-_-_-_-_-_-_-_-_-_- //
    // State carried between batches by the dense scheme. Under split ranges
    // both are implicit (slot x belongs to vertex x forever); dense ids have to
    // be told where the ranges currently end.
    int n_s = 0;                 // |V(H_i)|: node count of the carried BCG
    int n_b = 0;                 // blocks in it -- they own [0, 2*n_b), so
                                 // "M[x] is a block" is M[x] < 2*n_b
    int* d_L = NULL;             // minted id per extracted vertex   (numVert)
    int* d_mark_scan = NULL;     // exclusive scan of d_mark       (numVert + 1)

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ Forest Euler tour _-_-_-_-_-_-_-_-_-_-_-_-_- //
    // The spanning FOREST of a disconnected input gives c Euler circuits, one
    // per tree, whose list rankings each restart at zero. These carry the
    // per-component metadata that keeps their position ranges disjoint.
    //
    // A component of n_i vertices owns 2*n_i positions: 2*(n_i - 1) arcs plus
    // the two its root occupies. So an ISOLATED vertex still gets a 2-wide
    // block and a valid degenerate interval -- no special case needed.
    int  n_comp = 0;             // c, the number of components
    int* d_cid = NULL;           // node -> dense component index [0, c)
    int* d_croot = NULL;         // 1 iff this node is its tree's root
    int* d_csize = NULL;         // vertices per component
    int* d_abase = NULL;         // ARC-index offset per component      (scan of 2(n_i-1))
    int* d_pbase = NULL;         // POSITION offset per component       (scan of 2 n_i)
    int* d_slot = NULL;          // arc slot per non-root node (scan of "has a parent edge")
};

#endif // BCG_MEM_UTILS_H
