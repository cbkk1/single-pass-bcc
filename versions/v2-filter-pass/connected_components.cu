#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#include "utility/cuda_utility.cuh"
#include "extern_bcc/extern_bcc.cuh"       
#include "extern_bcc/bcg_memory_utils.cuh"


#define tb_size 1024

__device__ inline 
int find_compress(int i, int* temp_label) {
    int j = i;
    if (temp_label[j] == j) {
        return j;
    }
    do {
        j = temp_label[j];
    } while (temp_label[j] != j);

    int tmp;
    while((tmp = temp_label[i])>j) {
        temp_label[i] = j;
        i = tmp;
    }
    return j;
}


__device__ inline 
bool union_async(long idx, int src, int dst, int* temp_label) {
    // printf("Doing union_async for src: %d and dst: %d\n", src, dst);
    while(1) {
        int u = find_compress(src, temp_label);
        int v = find_compress(dst, temp_label);

        if(u == v) break;
        if(v > u) { int temp; temp = u; u = v; v = temp; }
        if(u == atomicCAS(&temp_label[u], u, v)) {
           return true;
        } 
    }
    return false;
}

__global__ 
void union_find_gpu_COO(
    int numVert, int* temp_label, 
    int* d_parent, uint64_t* d_sf_edges,
    int* d_first, int* d_last, 
    int* d_low, int* d_high) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        // The first group of n threads operates on d_parent.
        int u = idx;
        int v = d_parent[idx];

        if(u == v) return;

        int f_u = d_first[u];
        int l_u = d_last[u];

        int f_v = d_first[v];
        int l_v = d_last[v];
        
        int low_u = d_low[u];
        int high_u = d_high[u];
        
        int low_v = d_low[v];
        int high_v = d_high[v];
        
        // if it a fence edge; ignore, else do CC.
        if( (f_u <= low_v and l_u >= high_v) or (f_v <= low_u and l_v >= high_u) ) return;

        // printf("Calling union_async for tree edges src: %d and dst: %d\n", u, v);
        union_async(idx, u, v, temp_label);
        
    } else if (idx < 2 * numVert) {
        // The second group of n threads operates on d_sf_edges.
        int index = idx - numVert;
        if(d_sf_edges[index] == INT_MAX) {
            // printf("Returning for the cross-edges src: %ld\n", d_sf_edges[index]);    
            return;
        }
        // all these are cross-edges; so do CC for all
        int u = (d_sf_edges[index] >> 32) & 0xFFFFFFFF;
        int v = (d_sf_edges[index]) & 0xFFFFFFFF;
        // #ifdef DEBUG
        // printf("Calling union_async for cross-edges src: %d and dst: %d\n", u, v);
        // #endif
        union_async(idx, u, v, temp_label);
    }
}

__global__ 
void union_find_gpu_repair(
    int numVert, int* temp_label, int* d_counter,
    int* d_parent, uint64_t* d_sf_edges,
    int* d_first, int* d_last, 
    int* d_low, int* d_high) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        // The first group of n threads operates on d_parent.
        int u = idx;
        int v = d_parent[idx];

        if(u == v) return;

        int f_v = d_first[v];
        int l_v = d_last[v];
        
        int low_u = d_low[u];
        int high_u = d_high[u];
        
        // if it a fence edge; ignore, else do CC.
        if(f_v <= low_u and l_v >= high_u and d_counter[v] == 1) {
            // printf("u:%d, v:%d is a fence edge; counter[u]: %d and counter[v]:%d.\n", u, v, d_counter[u], d_counter[v]);
                return;
        }

        // printf("Calling union_async for tree edges src: %d and dst: %d\n", u, v);
        union_async(idx, u, v, temp_label);
        
    } else if (idx < 2 * numVert) {
        // The second group of n threads operates on d_sf_edges.
        int index = idx - numVert;
        if(d_sf_edges[index] == INT_MAX) {
            // printf("Returning for the cross-edges src: %ld\n", d_sf_edges[index]);    
            return;
        }

        // all these are cross-edges; so do CC for all
        int u = (d_sf_edges[index] >> 32) & 0xFFFFFFFF;
        int v = (d_sf_edges[index]) & 0xFFFFFFFF;
        // #ifdef DEBUG
        // printf("Calling union_async for cross-edges src: %d and dst: %d\n", u, v);
        // #endif
        union_async(idx, u, v, temp_label);
    }
}

//---------------------------------------------------------------------
// Fence-keep: the filter, moved INSIDE Fast_BCC.
//
// A block is ONE node here, so the de-merger gives a leaver a single link to
// the block it left. That link looks exactly like a bridge to the fence test
// (f_v <= low_u and l_v >= high_u both hold), so the test fires and cuts the
// leaver off from a remnant it is genuinely 2-connected to -- it WAS a member
// of that BCC. Compress-Graph's filter repairs this afterwards by unioning the
// beta classes at the block node. This kernel instead declines to fence the
// edge in the first place, which saves that whole pass.
//
//   mode 1 (blk) : never fence a tree edge with an endpoint < n_b_prev, i.e.
//                  any tree edge at a carried block node. Every component
//                  touching the block then fuses into one -- and cross edges
//                  are unioned unconditionally anyway -- so this is exactly
//                  what the filter does, one phase earlier.
//   mode 4 (down): never fence a tree edge whose PARENT is a carried block.
//                  Corrects mode 1. Un-fencing a tree edge (c,p) sets
//                  l[c]=l[p], and by the LCH contract l[c] is the class of
//                  c's parent edge (c,p) while l[p] is the class of p's
//                  parent edge -- so the merge happens AT THE PARENT, pairing
//                  its up-edge with that one down-edge. To fuse every edge at
//                  a block B the block must therefore be the parent: then
//                  every child c gets l[c]=l[B], and B's own up-edge needs no
//                  union because beta(B,par(B)) = l[deeper] = l[B] already.
//                  Mode 1 also un-fences B's up-edge (B as the child), whose
//                  merge lands at par(B) -- an own node, often a genuine cut
//                  vertex -- which is exactly the over-merge it suffers.
//   mode 2 (new) : exempt ONLY this round's leaver links: one endpoint a
//                  carried block (< n_b_prev), the other a freshly minted
//                  leaver id (>= n_s_prev, since dense_mint_kernel hands out
//                  ids from n_s upward). Strictly narrower than mode 1 -- it
//                  does nothing for a block's attachments carried in from an
//                  earlier round.
//---------------------------------------------------------------------
__global__
void union_find_gpu_keepfence(
    int numVert, int* temp_label,
    int* d_parent, uint64_t* d_sf_edges,
    int* d_first, int* d_last,
    int* d_low, int* d_high,
    int n_b_prev, int n_s_prev, int mode) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        int u = idx;
        int v = d_parent[idx];
        if (u == v) return;

        bool keep;
        if (mode == 1)      keep = (u < n_b_prev) || (v < n_b_prev);
        else if (mode == 4) keep = (v < n_b_prev);   // parent is a block
        else                keep = ((u < n_b_prev) && (v >= n_s_prev))
                                || ((v < n_b_prev) && (u >= n_s_prev));

        if (!keep) {
            int f_u = d_first[u], l_u = d_last[u];
            int f_v = d_first[v], l_v = d_last[v];
            int low_u = d_low[u], high_u = d_high[u];
            int low_v = d_low[v], high_v = d_high[v];
            // the unmodified fence test
            if ((f_u <= low_v and l_u >= high_v) or (f_v <= low_u and l_v >= high_u)) return;
        }
        union_async(idx, u, v, temp_label);

    } else if (idx < 2 * numVert) {
        int index = idx - numVert;
        if (d_sf_edges[index] == INT_MAX) return;
        // all these are cross-edges; so do CC for all
        int u = (d_sf_edges[index] >> 32) & 0xFFFFFFFF;
        int v = (d_sf_edges[index]) & 0xFFFFFFFF;
        union_async(idx, u, v, temp_label);
    }
}

// BCC_FENCE_KEEP=blk|new|down selects the kernel above; unset/0 keeps the stock
// fence test plus Compress-Graph's filter. Read once.
int bcc_fence_keep_mode() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("BCC_FENCE_KEEP");
        v = !e ? 0 : (e[0] == 'b' ? 1 : (e[0] == 'n' ? 2 :
                     (e[0] == 's' ? 3 : (e[0] == 'd' ? 4 : 0))));
    }
    return v;
}

// n_s before this round's de-merger minted the leaver ids; set in demerge_dense.
extern int g_bcc_n_s_prev;

// Literal reading of "G' <- PLAINEDGES(T) u Fc u D": run the stock CC, then
// add every de-merge link to the skeleton regardless of how the spanning tree
// classified it. A leaver link is (L[u], mu) with L[u] >= n_s_prev (freshly
// minted) and mu < n_b_prev (the block it left) -- dense_scatter_kernel.
// Selected by BCC_FENCE_KEEP=skel. This exists to check that the cheaper
// tree-edge-only exemption (mode "new") loses nothing: a de-merge link that is
// a BACK edge already forces its tree path non-fence, and one that is a CROSS
// edge is covered by the spanning forest Fc, so only the tree-edge case can
// actually be fenced away.
__global__
void union_demerge_links_kernel(const uint64_t* d_edges, long m, int* temp_label,
                                int n_b_prev, int n_s_prev) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t e = d_edges[i];
    int x = (int)(e >> 32), y = (int)(e & 0xFFFFFFFF);
    bool link = ((x < n_b_prev) && (y >= n_s_prev)) || ((y < n_b_prev) && (x >= n_s_prev));
    if (link) union_async(i, x, y, temp_label);
}

__global__ 
void cc_gpu(int* label, int* temp_label, int V) {       
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < V) {
        label[idx] = find_compress(idx, temp_label);
    }
}

__global__
void init_parent_label(int* temp_label, int* label, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < V) {
        temp_label[idx] = idx;
        label[idx] = idx;
    }
}


void CC(GPU_BCG& g_bcg_ds, const bool lastBatch) {

    uint64_t* d_sf_edges = g_bcg_ds.d_index;
    int V                = g_bcg_ds.numVert;

    int* d_parent           = g_bcg_ds.d_parent;

    int* d_first            = g_bcg_ds.d_first_occ;
    int* d_last             = g_bcg_ds.d_last_occ;

    int* d_low              = g_bcg_ds.d_first;
    int* d_high             = g_bcg_ds.d_last;

    int* temp_label         = g_bcg_ds.devRank;
    int* label              = g_bcg_ds.d_rep;

    int* d_counter          = g_bcg_ds.d_counter;

    int grid_size_final = CEIL(V, tb_size);

    init_parent_label<<<grid_size_final, tb_size>>>(
        temp_label, 
        label, 
        V);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init_parent_label_cc kernel");

    grid_size_final = CEIL(2*V, tb_size);
    const int fence_keep = bcc_fence_keep_mode();
    if(fence_keep == 3 && !lastBatch) {
        union_find_gpu_COO<<<grid_size_final, tb_size>>>(
            V, temp_label,
            d_parent, d_sf_edges,
            d_first, d_last,
            d_low, d_high);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize union_find_gpu_COO (skel)");
        if (g_bcg_ds.numEdges > 0) {
            union_demerge_links_kernel<<<CEIL(g_bcg_ds.numEdges, tb_size), tb_size>>>(
                g_bcg_ds.updated_edgelist, g_bcg_ds.numEdges, temp_label,
                g_bcg_ds.n_b, g_bcc_n_s_prev);
            CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize union_demerge_links kernel");
        }
    }
    else if(fence_keep && !lastBatch) {
        // g_bcg_ds.n_b is still LAST round's block count here: compress only
        // advances it after it returns, so ids below it are the carried blocks.
        union_find_gpu_keepfence<<<grid_size_final, tb_size>>>(
            V, temp_label,
            d_parent, d_sf_edges,
            d_first, d_last,
            d_low, d_high,
            g_bcg_ds.n_b, g_bcc_n_s_prev, fence_keep);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize union_find_gpu_keepfence kernel");
    }
    else if(!lastBatch) {
        union_find_gpu_COO<<<grid_size_final, tb_size>>>(
            V, temp_label, 
            d_parent, d_sf_edges,
            d_first, d_last,
            d_low, d_high);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize union_find_gpu_COO_cc kernel");
    }
    else {
        union_find_gpu_repair<<<grid_size_final, tb_size>>>(
            V, temp_label, d_counter, 
            d_parent, d_sf_edges,
            d_first, d_last,
            d_low, d_high);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize union_find_gpu_COO_cc kernel");        
    }

    grid_size_final = CEIL(V, tb_size);
    cc_gpu<<<grid_size_final, tb_size>>>(label, temp_label, V);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize cc_gpu_cc kernel");

    #ifdef DEBUG
        std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
        std::cout << "BCC Numbers from CC function:" << "\n";
        std::cout << "V from CC: " << V << std::endl;
        print_device_array(label, V);
        std::cout << "\nPrinting from CC complete" << std::endl;
        std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n" << std::endl;
    #endif
}
