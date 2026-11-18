#include <iostream>
#include <vector>
#include <set>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "extern_bcc/bcg_memory_utils.cuh"
#include "fast_bcc/list_ranking.cuh"
#include "utility/cuda_utility.cuh"

// #define DEBUG

// FOREST-CAPABLE. The original compacted with `thid - (thid > root)`, a closed
// form valid only when exactly ONE slot is empty. A spanning forest of c trees
// leaves c empty slots, so the slot comes from a prefix sum instead
// (d_slot, built in gpu_compress.cu), and the twin stride is E/2 = N - c.
__global__ 
void create_dup_edges(
    int *d_edges_to, 
    int *d_edges_from, 
    const uint64_t *d_edges_input, 
    const int *d_slot,
    const int *d_croot,
    int half,
    int N) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thid < N) {

        if (d_croot[thid])            // a root of its tree: no parent edge
          return;

        uint64_t i = d_edges_input[thid];

        int u = i >> 32;  // Extract higher 32 bits
        int v = i & 0xFFFFFFFF; // Extract lower 32 bits
        
        int k = d_slot[thid];
#ifdef BCC_GUARD_DUP
        if (u >= N || v >= N || u < 0 || v < 0 || k < 0 || k >= half)
            printf("[DUP] node %d (non-root) slot %d/%d parentEdge=%llx -> (%d,%d) N=%d\n",
                   thid, k, half, (unsigned long long)i, u, v, N);
#endif
        d_edges_from[k + half] = d_edges_to[k] = v;
        d_edges_to[k + half] = d_edges_from[k] = u;
    }
}


__global__
void update_first_last_nxt(int* d_edges_from, int* d_edges_to, int* d_first, int* d_last, int* d_next, uint64_t* d_index, int E) {
    
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thid < E) {
        int f = d_edges_from[d_index[thid]];
        // int t = d_edges_to[d_index[thid]];

        if (thid == 0) {
            d_first[f] = d_index[thid];
            return;
        }

        if(thid == E - 1) {
            d_last[f] = d_index[thid];
        }

        int pf = d_edges_from[d_index[thid - 1]];
        // int pt = d_edges_to[d_index[thid - 1]];

        // calculate the offset array
        if (f != pf) {
            d_first[f] = d_index[thid];
            // printf("d_last[%d] = d_index[%d] = %d\n", pf, thid - 1, d_index[thid - 1]);
            d_last[pf] = d_index[thid - 1];
        } else {
            d_next[d_index[thid - 1]] = d_index[thid];
        }
    }
}

__global__ 
void cal_succ(int* succ, const int* d_next, const int* d_first, const int* d_edges_from, int E) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < E) {
        int revEdge = (thid + E / 2) % E;

        if (d_next[revEdge] == -1) {
            succ[thid] = d_first[d_edges_from[revEdge]];
        } else {
            succ[thid] = d_next[revEdge];
        }
    }
}

__global__ 
void break_cycle_kernel(int *d_last, int *d_succ, int* d_roots, int roots_count, int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < roots_count) {
        int root = d_roots[idx];
        // printf("Root: %d\n", root);
        if (d_last[root] != -1) {
            int last_edge = d_last[root];
            int rev_edge = (last_edge + E / 2) % E;
            // printf("\nFor root: %d, last_edge: %d, rev_edge: %d\n", root, last_edge, rev_edge);
            // Set the successor of the last edge to point to itself
            d_succ[rev_edge] = -1;
        }
    }
}

// Shifts each component's locally-ranked arcs into its own slice of a shared
// index space. Applied twice: with the ARC offsets before compute_level (which
// scans over [0,E)), then by a further 2 per component for the POSITION space
// that compute_first_last_occ writes into.
__global__
void shift_ranks(int *rank, const int *d_edges_from, const int *d_cid,
                 const int *base, int per_comp, int E) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < E) {
        int c = d_cid[d_edges_from[tid]];
        rank[tid] += (base ? base[c] : 0) + per_comp * c;
    }
}

__global__
void find_parent(int E, int *rank, int *d_edges_to, int *d_edges_from, int *parent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < E) {
        int f = d_edges_from[tid];
        int t = d_edges_to[tid];
        int rev_edge = (tid + E / 2) % E;
        if(rank[tid] > rank[rev_edge]) {
            parent[t] = f;
        }
        else {
            parent[f] = t;
        }
    }
}

__global__ 
void merge_key_value(const int *arrayU, const int *arrayV, uint64_t *arrayE, uint64_t *d_indices, long size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Cast to int64_t to ensure the shift operates on 64 bits
        uint64_t u = arrayU[idx];
        uint64_t v = arrayV[idx];

        arrayE[idx] = (u << 32) | (v & 0xFFFFFFFFLL);

        d_indices[idx] = idx;
    }
}

__global__
void compute_level_kernel(
    int* devRank, int* d_edges_from, int* d_edges_to, 
    int* d_parent, int* devW1Sum, int N, int E) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < E) {
        int loc = E - 1 - devRank[idx];
        // idx is the edge number, so lets retrive the edge first
        int u = d_edges_from[idx];
        int v = d_edges_to[idx];
        
        if(d_parent[v] == u)
            devW1Sum[loc] = 1;
        else
            devW1Sum[loc] = -1;
    }
}

__global__
void finalise_level_kernel(
    int* d_edges_from, int* d_edges_to, 
    int* d_parent, 
    int* devRank, 
    int* d_prefix_sum, 
    int* d_level, 
    int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < E) {
        int loc = E - 1 - devRank[idx];
        int u = d_edges_from[idx];
        int v = d_edges_to[idx];

        // (p(V)，v)
        if(d_parent[v] == u) {
            d_level[v] = d_prefix_sum[loc];
        }
    }
}

void LexSortIndices(int* d_keys, int* d_values, uint64_t* d_indices_sorted, int num_items) {

    uint64_t *d_merged, *d_merged_keys_sorted;
    CUCHECK(cudaMalloc(&d_merged, sizeof(uint64_t) * num_items));
    CUCHECK(cudaMalloc(&d_merged_keys_sorted, sizeof(uint64_t) * num_items));

    uint64_t* d_indices;
    CUCHECK(cudaMalloc(&d_indices, sizeof(uint64_t)* num_items));   

    int blockSize = 1024;
    int numBlocks = (num_items + blockSize - 1) / blockSize; 

    // Initialize indices to 0, 1, 2, ..., num_items-1 also here
    merge_key_value<<<numBlocks, blockSize>>>(
        d_keys, 
        d_values, 
        d_merged, 
        d_indices, 
        num_items);
    CUCHECK(cudaDeviceSynchronize());

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_merged, d_merged_keys_sorted, d_indices, d_indices_sorted, num_items);
    CUCHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Sort indices based on keys
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_merged, d_merged_keys_sorted, d_indices, d_indices_sorted, num_items);

    cudaFree(d_merged);
    cudaFree(d_merged_keys_sorted);
    cudaFree(d_indices);
    cudaFree(d_temp_storage);
}

void compute_level(
    int* devRank, int* d_edges_from, int* d_edges_to, 
    int* d_parent, int* devW1Sum, int* d_level, int N, int E) {

    int blockSize = 1024;
    int numBlocks = (E + blockSize - 1) / blockSize;

    compute_level_kernel<<<numBlocks, blockSize>>>(
        devRank, 
        d_edges_from, 
        d_edges_to, 
        d_parent, 
        devW1Sum, 
        N, E);

    CUCHECK(cudaDeviceSynchronize());

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    int* d_level_info;
    CUCHECK(cudaMalloc(&d_level_info, sizeof(int)*E));

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, devW1Sum, d_level_info, E);
    // Allocate temporary storage
    CUCHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, devW1Sum, d_level_info, E);
    CUCHECK(cudaDeviceSynchronize());

    // devRank contains the prefix sum result
    finalise_level_kernel<<<numBlocks, blockSize>>>(
        d_edges_from, d_edges_to, 
        d_parent, devRank, d_level_info,
        d_level, E);
    CUCHECK(cudaDeviceSynchronize());

    CUCHECK(cudaFree(d_level_info));
    CUCHECK(cudaFree(d_temp_storage));
}

__global__
void compute_first_last_occ(
    int* d_edges_from, 
    int* d_edges_to, 
    int* d_first_occ, 
    int* d_last_occ, 
    int* d_level,
    int* devRank, 
    int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        int u = d_edges_to[idx];
        int v = d_edges_from[idx];       
        if (d_level[u] > d_level[v]) {
            d_last_occ[u] = devRank[idx]+1;
        } else {
            d_first_occ[v] = devRank[idx]+1;
        }
    }
}

__global__ 
// FOREST-CAPABLE. Every tree has its own root, and each root is seeded at both
// ends of its component's POSITION block [pbase, pbase + 2*n_i). A component of
// one vertex still gets a 2-wide block, so an isolated vertex ends up with a
// valid degenerate interval and needs no special case downstream.
__global__
void init_arrays(
    int *d_next, int *d_parent, int *d_level,
    int *d_first, int *d_last, 
    int* d_first_occ, int* d_last_occ, 
    const int* d_croot, const int* d_cid,
    const int* d_pbase, const int* d_csize,
    int E, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < E) {
        // Initialize d_next array
        d_next[idx] = -1;  // You can modify this initialization as per your requirement
    }

    if (idx < N) {
        // Initialize d_parent, d_first, and d_last arrays
        d_parent[idx] = idx;
        d_first[idx] = -1;
        d_last[idx] = -1;
        d_level[idx] = 0;

        if (d_croot[idx]) {
            int c = d_cid[idx];
            d_first_occ[idx] = d_pbase[c];
            d_last_occ[idx]  = d_pbase[c] + 2 * d_csize[c] - 1;
        } else {
            d_first_occ[idx] = 2 * N;
            d_last_occ[idx] = -1;
        }
    }
}

// FOREST-CAPABLE. `c` is the number of trees in the spanning forest; the
// per-component metadata (d_cid, d_croot, d_slot, d_abase, d_pbase, d_csize,
// d_roots) is prepared by forest_metadata() in gpu_compress.cu.
void cuda_euler_tour(
    int N, 
    int c,
    GPU_BCG& g_bcg_ds) {
    
    uint64_t* d_edges_input = g_bcg_ds.d_parentEdge; 
    int E = 2 * (N - c);          // N-c tree edges, two arcs each
    int roots_count = c;

    int *d_edges_to = g_bcg_ds.d_edges_to;
    int *d_edges_from = g_bcg_ds.d_edges_from;
    
    // index can be considered as edge_num
    uint64_t *d_index =     g_bcg_ds.d_index;
    int *d_next       =     g_bcg_ds.d_next;
    int *d_roots      =     g_bcg_ds.d_roots;
    int *d_parent     =     g_bcg_ds.d_parent;
    int *d_level      =     g_bcg_ds.d_level;
    int *d_first      =     g_bcg_ds.d_first;
    int *d_last       =     g_bcg_ds.d_last;
    int *d_first_occ  =     g_bcg_ds.d_first_occ;
    int *d_last_occ   =     g_bcg_ds.d_last_occ;

    // d_roots already holds all c roots, filled by forest_metadata().
    int blockSize = 1024;
    // init_arrays guards on BOTH `idx < E` and `idx < N`, so it must be launched
    // over max(E, N). With one tree E = 2N-2 > N and launching over E sufficed;
    // on a FOREST E = 2(N-c) can be far smaller than N, and every node past the
    // launch bound keeps garbage occurrences.
    int numBlocks = ((E > N ? E : N) + blockSize - 1) / blockSize;

    init_arrays<<<numBlocks, blockSize>>>(
        d_next, 
        d_parent,
        d_level, 
        d_first, 
        d_last,
        d_first_occ,
        d_last_occ,
        g_bcg_ds.d_croot, g_bcg_ds.d_cid,
        g_bcg_ds.d_pbase, g_bcg_ds.d_csize,
        E, N);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init_arrays kernel");

    // N, not N-1: the kernel guards on `thid < N` and skips roots itself, so it
    // needs the full [0, N) range. Launching N-1 threads drops node N-1 whenever
    // N-1 is an exact multiple of blockSize -- its tree edge never enters the
    // tour, the successor list closes into a cycle, and CudaSimpleListRank's
    // pointer-jumping loop then never terminates.
    numBlocks = (N + blockSize - 1) / blockSize;
    // Launch the kernel
    create_dup_edges<<<numBlocks, blockSize>>>(
        d_edges_to, 
        d_edges_from, 
        d_edges_input, 
        g_bcg_ds.d_slot,
        g_bcg_ds.d_croot,
        E / 2,
        N);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize create_dup_edges kernel");
    
    #ifdef DEBUG
        std::cout << "Printing from Euler Tour after creating duplicates:\n";
        DisplayDeviceEdgeList(d_edges_from, d_edges_to, E);
    #endif

    numBlocks = (E + blockSize - 1) / blockSize;

    LexSortIndices(d_edges_from, d_edges_to, d_index, E);

    #ifdef DEBUG
        std::cout << "Index array:\n";
        print_device_array(d_index, E);

        std::vector<int> sorted_from(E), sorted_to(E);
        std::vector<uint64_t> sorted_index(E);
        
        CUCHECK(cudaMemcpy(sorted_index.data(), d_index, sizeof(uint64_t) * E, cudaMemcpyDeviceToHost));
        CUCHECK(cudaMemcpy(sorted_from.data(), d_edges_from, sizeof(int) * E, cudaMemcpyDeviceToHost));
        CUCHECK(cudaMemcpy(sorted_to.data(), d_edges_to, sizeof(int) * E, cudaMemcpyDeviceToHost));

        // Print the sorted edges
        std::cout << "Sorted Edges:" << std::endl;
        for (int i = 0; i < E; ++i) {
            int idx = sorted_index[i];
            std::cout << i << ": (" << sorted_from[idx] << ", " << sorted_to[idx] << ")" << std::endl;
        }
    #endif

    update_first_last_nxt<<<numBlocks, blockSize>>>(
        d_edges_from, 
        d_edges_to, 
        d_first, 
        d_last, 
        d_next, 
        d_index, 
        E);

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_first_last_nxt");

    int *succ = g_bcg_ds.succ;
    int *devRank = g_bcg_ds.devRank;

    cal_succ<<<numBlocks, blockSize>>>(succ, d_next, d_first, d_edges_from, E);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize cal_succ");

    #ifdef DEBUG
        std::cout << "successor array before break_cycle_kernel:\n";
        print_device_array(succ, E);
    #endif

    // break cycle_kernel
    numBlocks = (roots_count + blockSize - 1) / blockSize;
    break_cycle_kernel<<<numBlocks, blockSize>>>(d_last, succ, d_roots, roots_count, E);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize break_cycle_kernel");

    CudaSimpleListRank(
        devRank, E, succ, 
        g_bcg_ds.notAllDone, g_bcg_ds.devRankNext, g_bcg_ds.devNotAllDone); // these 3 are auxilary ds

    // Each tree was ranked independently and restarts at 0, so the c rankings
    // collide. Shift each into its own ARC slice first -- compute_level reverses
    // the ranks and prefix-sums over [0,E), so it needs contiguous per-component
    // segments. Each tour is balanced (+1 down, -1 up), so the running sum
    // returns to zero at every boundary and each root still comes out at level 0.
    numBlocks = (E + blockSize - 1) / blockSize;
    shift_ranks<<<numBlocks, blockSize>>>(
        devRank, d_edges_from, g_bcg_ds.d_cid, g_bcg_ds.d_abase, 0, E);
    CUCHECK(cudaDeviceSynchronize());

    #ifdef DEBUG
        std::cout << "d_first array:\n";
        print_device_array(d_first, N);

        std::cout << "d_last array:\n";
        print_device_array(d_last, N);

        std::cout << "d_next array:\n";
        print_device_array(d_next, E);

        std::cout << "successor array:\n";
        print_device_array(succ, E);

        std::cout << "euler Path array:\n";
        print_device_array(devRank, E);
    #endif

    numBlocks = (E + blockSize - 1) / blockSize;
    find_parent<<<numBlocks, blockSize>>>(E, devRank, d_edges_to, d_edges_from, d_parent);
    CUCHECK(cudaDeviceSynchronize());

    if (getenv("BCC_INV")) {
        // O(N) structural check on the forest the tour just built: every
        // non-root must have been given a parent, and exactly c roots must be
        // marked. An "orphan" (non-root whose parent is itself) is what a
        // non-converged list ranking looks like from here -- see the companion
        // level check after compute_level.
        std::vector<int> pa(N), cr(N);
        CUCHECK(cudaMemcpy(pa.data(), d_parent,         N*4, cudaMemcpyDeviceToHost));
        CUCHECK(cudaMemcpy(cr.data(), g_bcg_ds.d_croot, N*4, cudaMemcpyDeviceToHost));
        int orphans = 0, orphan = -1, roots = 0;
        for (int v = 0; v < N; ++v) {
            roots += (cr[v] != 0);
            if (!cr[v] && pa[v] == v) { if (orphan < 0) orphan = v; ++orphans; }
        }
        if (orphans || roots != c)
            std::cout << "  [INV] N=" << N << " c=" << c << " E=" << E
                      << " roots_marked=" << roots << " orphans=" << orphans
                      << " first=" << orphan << "\n";
    }

    compute_level(devRank, d_edges_from, d_edges_to, d_parent, g_bcg_ds.devW1Sum, g_bcg_ds.d_level, N, E);

    if (getenv("BCC_INV")) {
        // The direct precursor of the a1[-1] crash: level[v] must be
        // level[parent[v]] + 1 for every non-root, and 0 at every root.
        std::vector<int> lv(N), pa(N), cr(N), cid(N);
        CUCHECK(cudaMemcpy(lv.data(), d_level,                N*4, cudaMemcpyDeviceToHost));
        CUCHECK(cudaMemcpy(pa.data(), d_parent,               N*4, cudaMemcpyDeviceToHost));
        CUCHECK(cudaMemcpy(cr.data(), g_bcg_ds.d_croot,       N*4, cudaMemcpyDeviceToHost));
        CUCHECK(cudaMemcpy(cid.data(), g_bcg_ds.d_cid,        N*4, cudaMemcpyDeviceToHost));
        int bad = 0, firstbad = -1;
        for (int v = 0; v < N; ++v) {
            bool ok = cr[v] ? (lv[v] == 0) : (pa[v] != v && lv[v] == lv[pa[v]] + 1);
            if (!ok) { if (firstbad < 0) firstbad = v; ++bad; }
        }
        if (bad) {
            std::cout << "  [LEVEL] N=" << N << " c=" << c << " E=" << E
                      << "  bad_levels=" << bad << " first=" << firstbad
                      << " (root=" << cr[firstbad] << " cid=" << cid[firstbad]
                      << " level=" << lv[firstbad] << " parent=" << pa[firstbad]
                      << " parent_level=" << lv[pa[firstbad]] << ")\n";
        }
    }
    
    #ifdef DEBUG
        std::cout << "Parent array:" << std::endl;
        print_device_array(d_parent, N);

        std::cout << "devW1Sum array:" << std::endl;
        print_device_array(g_bcg_ds.devW1Sum, E);

        std::cout << "d_level array:" << std::endl;
        print_device_array(g_bcg_ds.d_level, N);
    #endif


    //memset d_first_occ and d_last_occ
    //first_occ of root is 0
    //last_occ of root is E - 1

    // Now from ARC space into POSITION space. A component owns 2*n_i positions
    // but only 2*(n_i - 1) arcs; the extra 2 are the two its root occupies, so
    // the two offset tables differ by exactly 2 per component.
    numBlocks = (E + blockSize - 1) / blockSize;
    shift_ranks<<<numBlocks, blockSize>>>(
        devRank, d_edges_from, g_bcg_ds.d_cid, nullptr, 2, E);
    CUCHECK(cudaDeviceSynchronize());

    numBlocks = (E + blockSize - 1) / blockSize;
    compute_first_last_occ<<<numBlocks, blockSize>>>(
        d_edges_from, 
        d_edges_to, 
        d_first_occ, 
        d_last_occ, 
        g_bcg_ds.d_level,
        devRank, 
        E);
    CUCHECK(cudaDeviceSynchronize());

    if(g_verbose) {
        std::cout << "Parent array:" << std::endl;
        print_device_array(d_parent, N);

        std::cout << "d_level array:" << std::endl;
        print_device_array(g_bcg_ds.d_level, N);
        
        std::cout << "First Occurrence array:" << std::endl;
        print_device_array(d_first_occ, N);

        std::cout << "Last Occurrence array:" << std::endl;
        print_device_array(d_last_occ, N);
    }
    
}