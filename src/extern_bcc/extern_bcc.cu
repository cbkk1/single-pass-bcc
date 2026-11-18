//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <omp.h>
#include <vector>
#include <cassert>
#include <numeric>
#include <iterator>
#include <iostream>
#include <functional>
#include <cstdlib>

//---------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------
#include "utility/timer.hpp"             
#include "utility/utility.hpp"           
#include "utility/cuda_utility.cuh"      

//---------------------------------------------------------------------
// Spanning Tree and Biconnected Components (BCC) related functions
//---------------------------------------------------------------------
#include "extern_bcc/bcg_memory_utils.cuh"
#include "extern_bcc/edge_cleanup.cuh"
#include "extern_bcc/extern_bcc.cuh"
#include "extern_bcc/single_pass.cuh"
#include "extern_bcc/serial_bcc.cuh"
#include "extern_bcc/lch.cuh"

// Set from -g / --gpu-bcc in main.cu: run Algorithm 3 with dense ids and
// Compress-Graph on the GPU, instead of split ranges with the host Tarjan.
extern bool g_gpu_bcc;

// #define DEBUG

 /*
  ____                                ____        _       _     
 |  _ \ _ __ ___   ___ ___  ___ ___  | __ )  __ _| |_ ___| |__  
 | |_) | '__/ _ \ / __/ _ \/ __/ __| |  _ \ / _` | __/ __| '_ \ 
 |  __/| | | (_) | (_|  __/\__ \__ \ | |_) | (_| | || (__| | | |
 |_|   |_|  \___/ \___\___||___/___/ |____/ \__,_|\__\___|_| |_|
                                                                
*/

/*
  _   _           _       _         __  __                   _             
 | | | |_ __   __| | __ _| |_ ___  |  \/  | __ _ _ __  _ __ (_)_ __   __ _ 
 | | | | '_ \ / _` |/ _` | __/ _ \ | |\/| |/ _` | '_ \| '_ \| | '_ \ / _` |
 | |_| | |_) | (_| | (_| | ||  __/ | |  | | (_| | |_) | |_) | | | | | (_| |
  \___/| .__/ \__,_|\__,_|\__\___| |_|  |_|\__,_| .__/| .__/|_|_| |_|\__, |
       |_|                                      |_|   |_|            |___/ 
*/

void construct_BCG(
    GPU_BCG& g_bcg_ds,
    const size_t BatchSize,
    const bool isLastBatch,
    TimerManager& timer_manager) {

    auto start = std::chrono::high_resolution_clock::now();

    // Single-pass: phase 3 (de-merge) -> phase 4 (BCC) -> phase 5 (compose).
    // There is no transform step and no pass-1 spanning tree any more.
    single_pass_batch(g_bcg_ds, (long)BatchSize);

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    timer_manager.add_function_time("Single-pass batch", dur);
}

int extern_bcc(GPU_BCG& g_bcg_ds) {
    // Single-pass: no pass-1 spanning tree. That pass cost a full read of the
    // edge stream, and dropping it is the point of the algorithm. The active
    // edge set therefore starts EMPTY rather than holding a tree of n-1 edges.
    Timer myTimer;
    auto dur = 0L;

    TimerManager timer_manager;

    g_bcg_ds.numEdges = 0;

    // -------------------- Assigning to local variables -------------------- //
    const uint64_t* h_edgelist  =   g_bcg_ds.h_edgelist;
    uint64_t* d_edgelist        =   g_bcg_ds.d_edgelist;

    int nodes = g_bcg_ds.numVert;
    long edges = g_bcg_ds.orig_numEdges;
    long batch_size = g_bcg_ds.batchSize;

    int original_num_verts = g_bcg_ds.orig_numVert;

    // -------------------- *************************** -------------------- //

    int num_batches = edges / batch_size;
    if (edges % batch_size != 0) {
        num_batches++;
    }

    myTimer.reset();

    auto total = 0;

    for (int i = 0; i < num_batches; i++) {
        auto copy_time_start = std::chrono::high_resolution_clock::now();
        
        long start = i * batch_size;
        long end = std::min((i + 1) * batch_size, edges);
        long num_elements_in_batch = end - start;    

        CUDA_CHECK(cudaMemcpy(
            d_edgelist, 
            h_edgelist + start, 
            num_elements_in_batch * sizeof(uint64_t), 
            cudaMemcpyHostToDevice), 
        "Memcpy error");

        if(g_verbose) {
            std::cout << "\n\n\t******************************************\n";
            std::cout << "\t***          Batch " << i << ": " << start << " to " << end << "          ***\n";
            std::cout << "\tNumber of elements in batch: " << num_elements_in_batch << "\n";
            std::cout << "\t******************************************\n\n" << std::endl;
        }

        auto copy_time_end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(copy_time_end - copy_time_start).count();

        total += dur;

        bool isLastBatch = (i == num_batches - 1);

        if (g_gpu_bcc) single_pass_batch_gpu(g_bcg_ds, num_elements_in_batch);
        else            construct_BCG(g_bcg_ds, num_elements_in_batch, isLastBatch, timer_manager);

        // BCC_PREFIX=1: after every batch, compare the count derivable from the
        // current BCG against a serial pass over the prefix seen so far. The
        // first batch that diverges is the one that lost something.
        if (g_gpu_bcc && getenv("BCC_PREFIX")) {
            const int two_nb = g_bcg_ds.n_b;
            long br = 0;
            if (g_bcg_ds.numEdges > 0) {
                std::vector<uint64_t> fin(g_bcg_ds.numEdges);
                CUDA_CHECK(cudaMemcpy(fin.data(), g_bcg_ds.updated_edgelist,
                                      g_bcg_ds.numEdges * sizeof(uint64_t), cudaMemcpyDeviceToHost), "prefix: edges");
                for (uint64_t e : fin) {
                    int u = (int)(e >> 32), v = (int)(e & 0xFFFFFFFF);
                    if (u >= two_nb && v >= two_nb) ++br;
                }
            }
            long got = g_bcg_ds.n_b + br;
            long want = serial_bcc_count(g_bcg_ds.h_edgelist, end, nodes, nullptr, nullptr);
            std::cout << "  [prefix] batch " << i << " edges<=" << end
                      << "  got " << got << "  want " << want
                      << (got == want ? "" : "   <<<< DIVERGES") << "\n";
        }
    }

    // std::cout << "Total Copy Times: " << total << " ms." << "\n" << std::endl;

    #ifdef DEBUG
        std::cout << "Mapping Array:" << "\n";
        print_device_array(g_bcg_ds.d_mapping, nodes);
    #endif

    auto new_dur = myTimer.stop();
    std::cout <<"computeBCG finished in: " << dur + new_dur - total <<" ms." << "\n";

    timer_manager.print_times();

    // The deliverable is the supernode mapping M. Count the distinct
    // supernodes original vertices land in.
    std::vector<int> host_map(original_num_verts);
    CUDA_CHECK(cudaMemcpy(host_map.data(), g_bcg_ds.d_mapping,
                          original_num_verts * sizeof(int), cudaMemcpyDeviceToHost),
               "Unable to copy back mapping array");

    std::set<int> supernodes(host_map.begin(), host_map.end());

    // ---- the biconnected-component count ----------------------------------
    // With the stream exhausted, the active edge set IS the block-cut forest,
    // so every biconnected component is either
    //
    //   a block   -- a node pair with ids >= n, counted by its primary half, or
    //   a bridge  -- an active edge, deliberately left uncompressed.
    //
    // A block owns an aligned PAIR of ids, so only the primary (even offset
    // from n) is counted; the second half is the same component.
    //
    // Both sources are needed for the block set: a block that absorbed nobody
    // (all its members are cut vertices) shows up only in the edge list, and a
    // block that is a whole connected component has no incident edge at all and
    // shows up only in M.
    // Under dense ids the readout is simpler and needs no scan of M: a block
    // owns an aligned pair in [0, 2*n_b) and everything that kept its own
    // identity owns an id above that, so a bridge is exactly an edge with both
    // endpoints >= 2*n_b -- every other edge is a block's own pair edge or an
    // attachment to one.
    if (g_gpu_bcc) {
        const int two_nb = g_bcg_ds.n_b;
        long dense_bridges = 0;
        std::vector<uint64_t> fin(g_bcg_ds.numEdges);
        if (g_bcg_ds.numEdges > 0) {
            CUDA_CHECK(cudaMemcpy(fin.data(), g_bcg_ds.updated_edgelist,
                                  g_bcg_ds.numEdges * sizeof(uint64_t), cudaMemcpyDeviceToHost),
                       "Unable to copy back the final edge list");
            for (uint64_t e : fin) {
                int u = (int)(e >> 32), v = (int)(e & 0xFFFFFFFF);
                if (u >= two_nb && v >= two_nb) ++dense_bridges;
            }
        }
        const long dense_total = g_bcg_ds.n_b + dense_bridges;

        // LCH readout -- the output format of the paper's Algorithm 1:
        //     BCC(j) = { v : label[v] = j }  u  { componentHead[j] }
        // One host pass over the finished block-cut graph; M and the edge set
        // are already here for the count, so this moves no extra data.
        std::vector<int> label, head;
        const int n_comp = compute_lch(host_map, original_num_verts, fin,
                                       g_bcg_ds.n_s, g_bcg_ds.n_b, label, head);
        long labelled = 0, heads = 0;
        for (int v = 0; v < original_num_verts; ++v) if (label[v] >= 0) ++labelled;
        for (int j = 0; j < n_comp; ++j)             if (head[j]  >= 0) ++heads;
        // How many BCCs does a single vertex head? A cut vertex in k bccs heads
        // k-1 of them, so this is the cut-vertex multiplicity distribution.
        std::vector<int> headmult(original_num_verts, 0);
        for (int j = 0; j < n_comp; ++j)
            if (head[j] >= 0 && head[j] < original_num_verts) ++headmult[head[j]];
        long distinct_heads = 0, max_mult = 0; int argmax = -1;
        std::vector<long> hist(8, 0);
        for (int v = 0; v < original_num_verts; ++v) {
            int k = headmult[v];
            if (k > 0) ++distinct_heads;
            if (k > max_mult) { max_mult = k; argmax = v; }
            if (k > 0) ++hist[std::min<long>(k, 7)];
        }

        std::cout << "\n========================================\n"
                  << "       Biconnected components\n"
                  << "========================================\n\n"
                  << "Biconnected components:   " << dense_total        << "\n"
                  << "  of which bridges:       " << dense_bridges      << "\n"
                  << "  2-connected blocks:     " << g_bcg_ds.n_b       << "\n"
                  << "Supernodes in M:          " << supernodes.size()  << "\n"
                  << "BCG nodes (n_s):          " << g_bcg_ds.n_s       << "\n"
                  << "----------------------------------------\n"
                  << "LCH output\n"
                  << "  components (labels):    " << n_comp             << "\n"
                  << "  vertices labelled:      " << labelled           << "\n"
                  << "  components with a head: " << heads << "\n"
                  << "  distinct head vertices:  " << distinct_heads << "\n"
                  << "  max BCCs headed by one v: " << max_mult
                  << "  (vertex " << argmax << ")\n"
                  << "  heads of 1/2/3/4/5/6/7+ : " << hist[1] << " " << hist[2]
                  << " " << hist[3] << " " << hist[4] << " " << hist[5]
                  << " " << hist[6] << " " << hist[7]              << "\n"
                  << "========================================\n";
        return (int)dense_total;
    }

    auto is_primary = [&](int id) {
        return id >= original_num_verts && ((id - original_num_verts) % 2 == 0);
    };

    std::set<int> blocks;
    for (int s : host_map) if (is_primary(s)) blocks.insert(s);

    long bridges = 0;
    if (g_bcg_ds.numEdges > 0) {
        std::vector<uint64_t> final_edges(g_bcg_ds.numEdges);
        CUDA_CHECK(cudaMemcpy(final_edges.data(), g_bcg_ds.updated_edgelist,
                              g_bcg_ds.numEdges * sizeof(uint64_t), cudaMemcpyDeviceToHost),
                   "Unable to copy back the final edge list");
        for (uint64_t e : final_edges) {
            int u = (int)(e >> 32), v = (int)(e & 0xFFFFFFFF);
            if (is_primary(u)) blocks.insert(u);
            if (is_primary(v)) blocks.insert(v);
        }
        bridges = bcg_bridge_count(final_edges.data(), (long)final_edges.size(),
                                   original_num_verts, g_bcg_ds.V_max);
    }

    std::cout << "\n========================================\n"
              << "       Biconnected components\n"
              << "========================================\n\n"
              << "Biconnected components:   " << (blocks.size() + bridges) << "\n"
              << "  of which bridges:       " << bridges       << "\n"
              << "  2-connected blocks:     " << blocks.size() << "\n"
              << "Supernodes in M:          " << supernodes.size() << "\n"
              << "========================================\n";

    return (int)(blocks.size() + bridges);
}

