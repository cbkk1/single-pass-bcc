//---------------------------------------------------------------------
// CUDA & CUB Libraries
//---------------------------------------------------------------------
#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// CUDA Utility & helper functions
//---------------------------------------------------------------------
#include "utility/timer.hpp"
#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"
#include <cstdlib>

extern int local_block_size;

#define DEBUG

__global__
void init_bcc_num_kernel(int* d_mapping, int* d_imp_bcc_num, int numVert) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numVert) {
        d_imp_bcc_num[idx] = idx;
        d_mapping[idx] = idx;
    }
}

void print_mem_info() {
    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte), "Error: cudaMemGetInfo fails");

    double free_db = static_cast<double>(free_byte); 
    double total_db = static_cast<double>(total_byte);
    double used_db = total_db - free_db;
    std::cout << "----------------------------------------\n"
          << "GPU Memory Usage Post-Allocation:\n"
          << "Used:     " << used_db / (1024.0 * 1024.0) << " MB\n"
          << "Free:     " << free_db / (1024.0 * 1024.0) << " MB\n"
          << "Total:    " << total_db / (1024.0 * 1024.0) << " MB\n"
          << "========================================\n\n";
}

GPU_BCG::GPU_BCG(int vertices, long num_edges, long _batchSize) : numVert(vertices), orig_numVert(vertices), orig_numEdges(num_edges), batchSize(_batchSize) {

    // Phase 3 emits the batch edge itself plus two edges per extracted endpoint
    // (a leaver joins both halves of its block), so at most 5 per batch edge.
    // The carried block-cut graph has one edge per block pair plus two per
    // block/cut-vertex incidence, so it stays O(numVert).
    // These three coefficients were sized for the ALIGNED-PAIR scheme, in which
    // a block owned two ids: hence "a leaver joins both halves of its block"
    // (5 edges per batch edge), "one edge per block pair" (6n) and "blocks take
    // an aligned pair" (2n). Blocks have been one node since the no-pair
    // redesign, and the sizing was never revisited.
    //
    // The bounds the current design actually supports:
    //   A_B = 3   a leaver emits ONE de-merge link, so 1 + 2 endpoints
    //   A_N = 1.5 H_i is a FOREST, so |E(H_i)| < n_s <= n
    //   F_N = 1.0 n_s starts at n (M[v]=v) and compress-graph is monotone
    //             non-increasing (n_s' = n_b' + n_o' <= n_t), so peak
    //             n_t <= n + 2b, which the +2*batchSize below already covers
    //
    // Validated 2026-11-17 over 29 graphs x {64k, 500k}: 58/58 cells exact
    // against serial Tarjan, zero asserts, runtime unchanged, and 49% less
    // device memory (496 GB -> 252 GB summed). Both kmer graphs went from
    // out-of-memory to correct. Measured n_s/n over those graphs is 0.06-0.46,
    // so even F_N = 1.0 keeps 2x headroom on every graph tested.
    //
    // Still env-tunable: the asserts in gpu_compress.cu and single_pass.cu are
    // the safety net, and a graph outside the tested set that needs more can be
    // run with BCC_ALLOT_B / BCC_ALLOT_N / BCC_VFB_N raised. Note the failure
    // mode of too-small F_N is a WRONG ANSWER, not a crash, so anything that
    // lowers these further must be re-validated with -c, not just run.
    auto envd = [](const char* k, double d) {
        const char* v = getenv(k); return v ? atof(v) : d; };
    const double A_B = envd("BCC_ALLOT_B", 3.0);
    const double A_N = envd("BCC_ALLOT_N", 1.5);
    const double F_N = envd("BCC_VFB_N",   1.0);
    max_allot = (long)(A_B * batchSize) + (long)(A_N * numVert);

    // The edge buffers carry headroom past max_allot, because Compress-Graph
    // both grows the list before handing it to Fast_BCC and grows it again on
    // the way out:
    //
    //   + V_fb        one super-root edge per component of the transient graph
    //   x 2           the rebuild emits 2 edges per (own-node, block) incidence,
    //                 so an edge internal to a block with BOTH ends cut costs 4
    //                 while one with both ends absorbed costs 0. The 2x is an
    //                 empirical bound, not a proof -- such edges measured ~8.5%
    //                 of the BCG on web-BerkStan -- so the asserts in
    //                 gpu_compress.cu are load-bearing, and the dedup right
    //                 afterwards collapses the result back to O(n_s).
    numEdges = 2 * max_allot + 2L * numVert + 2 * batchSize + 8;
    edge_capacity = numEdges;
    E = numEdges * 2;

    // LEGACY split-range path only (no -g). Ids are split: original vertices
    // own [0, n), blocks own [n, 3n). There a block takes an ALIGNED PAIR of ids -- one node cannot express the
    // 2-connection between a block and its single attachment point. Blocks are
    // edge-disjoint and span >= 3 vertices each, so there are at most (n-1)/2
    // of them and n ids would in principle do; [n, 3n) leaves headroom for the
    // extra blocks that batch-boundary splits can transiently create.
    V_max = 3 * numVert;

    // Ceiling on the transient node count n_t that the BCC step ever sees.
    // Fast_BCC runs on the transient graph, so every one of its arrays is sized
    // on this, not on numVert. See the field comment in bcg_memory_utils.cuh.
    V_fb = (int)(F_N * numVert) + 2 * static_cast<int>(batchSize) + 2;

    // Node-indexed arrays that BOTH paths touch (d_rep above all: the split-range
    // path uploads V_max labels into it, the dense path uses it as Fast_BCC's
    // label array over V_fb nodes) have to hold whichever is larger.
    const int V_alloc = std::max(V_max, V_fb);


    Timer myTimer;

    /* ------------------------------ Batch Spanning Tree ds starts ------------------------------ */

    std::cout << "batchSize: " << batchSize << std::endl;
    
    size_t batch_alloc = batchSize * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&d_edgelist, batch_alloc),       "Allocation error");
    CUDA_CHECK(cudaMalloc(&d_parentEdge, sizeof(uint64_t) * V_fb),       "Failed to allocate memory for d_parentEdge");
    CUDA_CHECK(cudaMalloc(&d_rep, sizeof(int) * V_alloc),                     "Failed to allocate memory for d_rep_hook");
    CUDA_CHECK(cudaMalloc((void **)&d_flag, sizeof(int)),                   "Failed to allocate memory for c_flag");
    /* ------------------------------ Spanning Tree ds ends ------------------------------ */

    /* ------------------------------ BCC ds starts ------------------------------ */
    CUDA_CHECK(cudaMalloc((void**)&updated_edgelist, numEdges * sizeof(uint64_t)), "Failed to allocate original_edgelist array");
    CUDA_CHECK(cudaMalloc((void**)&d_edge_buffer,    numEdges * sizeof(uint64_t)), "Failed to allocate d_edge_buffer array");

    CUDA_CHECK(cudaMalloc((void**)&d_mapping,        numVert * sizeof(int)), "Failed to allocate d_mapping array");
    CUDA_CHECK(cudaMalloc((void**)&d_imp_bcc_num,    numVert * sizeof(int)), "Failed to allocate d_imp_bcc_num array");

    // set these two arrays (one time work)
    long threadsPerBlock = maxThreadsPerBlock;
    size_t blocks = (numVert + threadsPerBlock - 1) / threadsPerBlock;

    // Becoz zeroth batch is a spanning tree and in a spanning every vertex is a cut vertex and ibcc
    // number of each vertex is the number itself
    init_bcc_num_kernel<<<blocks, threadsPerBlock>>>(d_mapping, d_imp_bcc_num, numVert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to set bcc numbers kernel");

    /* ------------------------------ step 1: Eulerian Tour ds starts ------------------------------ */

    // Fast_BCC's Euler tour needs 2N-2 slots for an N-node tree; N is V_fb here.
    long euler_edges = 2L * V_fb;

    CUDA_CHECK(cudaMalloc((void **)&d_first, sizeof(int) * V_fb), "Failed to allocate d_first");
    CUDA_CHECK(cudaMalloc((void **)&d_last,  sizeof(int) * V_fb),  "Failed to allocate d_last");

    CUDA_CHECK(cudaMalloc((void **)&d_edges_to,     sizeof(int) * euler_edges), "Failed to allocate d_edges_to");
    CUDA_CHECK(cudaMalloc((void **)&d_edges_from,   sizeof(int) * euler_edges), "Failed to allocate d_edges_from");
    CUDA_CHECK(cudaMalloc((void **)&d_next,         sizeof(int) * euler_edges), "Failed to allocate memory for d_next");
    CUDA_CHECK(cudaMalloc((void **)&succ,           sizeof(int) * euler_edges), "Failed to allocate succ array");
    CUDA_CHECK(cudaMalloc((void **)&devRank,        sizeof(int) * euler_edges), "Failed to allocate devRank array");
    CUDA_CHECK(cudaMalloc((void **)&devW1Sum,       sizeof(int) * euler_edges), "Failed to allocate devW1Sum");

    CUDA_CHECK(cudaMalloc((void **)&d_index, sizeof(uint64_t) * euler_edges), "Failed to allocate memory for d_index");

    CUDA_CHECK(cudaMalloc((void **)&d_roots, sizeof(int) * V_fb), "Failed to allocate memory for d_roots");
    // List Ranking Params
    CUDA_CHECK(cudaMallocHost((void **)&notAllDone, sizeof(int)), "Failed to allocate notAllDone");
    CUDA_CHECK(cudaMalloc((void **)&devRankNext, sizeof(ull) * euler_edges), "Failed to allocate devRankNext");
    CUDA_CHECK(cudaMalloc((void **)&devNotAllDone, sizeof(int)), "Failed to allocate devNotAllDone");
    // List Ranking Params end

    // output arrays
    CUDA_CHECK(cudaMalloc((void **)&d_parent,       sizeof(int) * V_fb), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMalloc((void **)&d_level,        sizeof(int) * V_fb),  "Failed to allocate d_level");
    CUDA_CHECK(cudaMalloc((void **)&d_first_occ,    sizeof(int) * V_fb), "Failed to allocate memory for d_first_occ");
    CUDA_CHECK(cudaMalloc((void **)&d_last_occ,     sizeof(int) * V_fb), "Failed to allocate memory for d_last_occ");
    /* ------------------------------ Eulerian Tour ds ends ------------------------------ */

    /* ------------------------------------ step 2: BCC ds starts --------------------------------------- */
    
    CUDA_CHECK(cudaMalloc(&d_left,  sizeof(int) * V_fb), "Failed to allocate memory for d_left");
    CUDA_CHECK(cudaMalloc(&d_right, sizeof(int) * V_fb), "Failed to allocate memory for d_right");

    long n_asize = (2L * V_fb + local_block_size - 1) / local_block_size;

    CUDA_CHECK(cudaMalloc(&d_na1, n_asize * sizeof(int)) , "Failed to allocate memory to d_na1");
    CUDA_CHECK(cudaMalloc(&d_na2, n_asize * sizeof(int)), "Failed to allocate memory to d_na1");
    

    // For updating bcc numbers
    CUDA_CHECK(cudaMalloc(&d_bcc_flag, sizeof(int) * V_fb), "Failed to allocate memory for d_bcc_flag");
    CUDA_CHECK(cudaMalloc(&iscutVertex, sizeof(int) * V_alloc), "Failed to allocate memory for iscutVertex");

    // This is for removing self loops and duplicates
    CUDA_CHECK(cudaMalloc((void**)&d_flags, numEdges * sizeof(unsigned char)), "Failed to allocate flag array");
    CUDA_CHECK(cudaMallocHost(&h_num_selected_out, sizeof(long)),   "Failed to allocate pinned memory for h_num_items value");
    CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(long)), "Failed to allocate d_num_selected_out");

    // Repair data-structure
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int) * V_alloc),   "Failed to allocate memory for d_counter");
    /* ------------------------------------ BCC ds ends --------------------------------------- */

    /* ------------------------------ Single-pass de-merger ------------------------------ */
    // d_mark is indexed by ORIGINAL vertex, so numVert is the right size; it is
    // only the values in d_mapping that reach up into [numVert, V_max).
    CUDA_CHECK(cudaMalloc(&d_mark, sizeof(int) * numVert), "Failed to allocate d_mark");
    CUDA_CHECK(cudaMalloc(&d_edge_out_count,  sizeof(int) * (batchSize + 1)), "Failed to allocate d_edge_out_count");
    CUDA_CHECK(cudaMalloc(&d_edge_out_offset, sizeof(int) * (batchSize + 1)), "Failed to allocate d_edge_out_offset");

    /* ------------------------------ GPU Compress-Graph ------------------------------ */
    // Node-indexed, so V_fb: these live over the TRANSIENT graph. One slot past
    // V_fb on the scans, because an exclusive scan of n items needs n+1 to hold
    // the total.
    CUDA_CHECK(cudaMalloc(&d_cc,         sizeof(int) * V_fb),       "Failed to allocate d_cc");
    CUDA_CHECK(cudaMalloc(&d_blk_edges,  sizeof(int) * V_fb),       "Failed to allocate d_blk_edges");
    CUDA_CHECK(cudaMalloc(&d_blk_flag,   sizeof(int) * (V_fb + 1)), "Failed to allocate d_blk_flag");
    CUDA_CHECK(cudaMalloc(&d_blk_struct, sizeof(int) * V_fb),       "Failed to allocate d_blk_struct");
    CUDA_CHECK(cudaMalloc(&d_first_beta, sizeof(int) * V_fb),       "Failed to allocate d_first_beta");
    CUDA_CHECK(cudaMalloc(&d_own,        sizeof(int) * V_fb),       "Failed to allocate d_own");
    CUDA_CHECK(cudaMalloc(&d_scan_b,     sizeof(int) * (V_fb + 1)), "Failed to allocate d_scan_b");
    CUDA_CHECK(cudaMalloc(&d_scan_o,     sizeof(int) * (V_fb + 1)), "Failed to allocate d_scan_o");
    CUDA_CHECK(cudaMalloc(&d_newId,      sizeof(int) * V_fb),       "Failed to allocate d_newId");

    // Edge-indexed.
    CUDA_CHECK(cudaMalloc(&d_beta, sizeof(int) * numEdges),       "Failed to allocate d_beta");
    CUDA_CHECK(cudaMalloc(&d_cnt,  sizeof(int) * (numEdges + 1)), "Failed to allocate d_cnt");
    CUDA_CHECK(cudaMalloc(&d_off,  sizeof(int) * (numEdges + 1)), "Failed to allocate d_off");
    // Dense-id de-merger. Both indexed by ORIGINAL vertex.
    CUDA_CHECK(cudaMalloc(&d_L,         sizeof(int) * numVert),       "Failed to allocate d_L");
    CUDA_CHECK(cudaMalloc(&d_mark_scan, sizeof(int) * (numVert + 1)), "Failed to allocate d_mark_scan");

    // Forest Euler tour metadata. Node-indexed, so V_fb; one slot past for the
    // scans' totals.
    CUDA_CHECK(cudaMalloc(&d_cid,   sizeof(int) * V_fb),       "Failed to allocate d_cid");
    CUDA_CHECK(cudaMalloc(&d_croot, sizeof(int) * V_fb),       "Failed to allocate d_croot");
    CUDA_CHECK(cudaMalloc(&d_csize, sizeof(int) * (V_fb + 1)), "Failed to allocate d_csize");
    CUDA_CHECK(cudaMalloc(&d_abase, sizeof(int) * (V_fb + 1)), "Failed to allocate d_abase");
    CUDA_CHECK(cudaMalloc(&d_pbase, sizeof(int) * (V_fb + 1)), "Failed to allocate d_pbase");
    CUDA_CHECK(cudaMalloc(&d_slot,  sizeof(int) * (V_fb + 1)), "Failed to allocate d_slot");

    // The dense scheme starts with every vertex its own node and no blocks.
    n_s = numVert;
    n_b = 0;
    /* ------------------------------------ Compress-Graph ends --------------------------------------- */

    /* ------------------------------------ Single-pass ends --------------------------------------- */

    // pinned memories
    CUDA_CHECK(cudaMallocHost(&h_max_ps_bcc, sizeof(int)),                    "Failed to allocate pinned memory for max_ps_bcc value");
    CUDA_CHECK(cudaMallocHost(&h_max_ps_cut_vertex, sizeof(int)),             "Failed to allocate pinned memory for max_ps_cut_vertex value");

    // Initialize GPU memory
    auto dur = myTimer.stop();

    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte), "Error: cudaMemGetInfo fails");

    double free_db = static_cast<double>(free_byte); 
    double total_db = static_cast<double>(total_byte);
    double used_db = total_db - free_db;

    std::cout << "========================================\n"
          << "       Allocation Details & GPU Memory Usage \n"
          << "========================================\n\n"
          << "Vertices (numVert):             " << numVert << "\n"
          << "Max supernode ids (V_max):      " << V_max << "\n"
          << "Fast_BCC node ceiling (V_fb):   " << V_fb << "\n"
          << "Edges (numEdges):               " << numEdges << "\n"
          << "Max Allotment:                  " << max_allot << "\n"
          << "Device Allocation & Setup Time: " << dur << " ms\n"
          << "Batch Size:                     " << batchSize << "\n"
          << "Total Number of Batches:        " << (orig_numEdges + batchSize - 1) / batchSize << "\n" 
          << "----------------------------------------\n"
          << "GPU Memory Usage Post-Allocation:\n"
          << "Used:     " << used_db / (1024.0 * 1024.0) << " MB\n"
          << "Free:     " << free_db / (1024.0 * 1024.0) << " MB\n"
          << "Total:    " << total_db / (1024.0 * 1024.0) << " MB\n"
          << "========================================\n\n";
}