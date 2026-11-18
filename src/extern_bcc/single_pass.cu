//---------------------------------------------------------------------
// Single-pass (transformation-free) BCC: the de-merger.
//---------------------------------------------------------------------

#include <cassert>
#include <cstdlib>
#include <iostream>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <vector>

#include "utility/cuda_utility.cuh"
#include "extern_bcc/single_pass.cuh"
#include "extern_bcc/serial_bcc.cuh"
#include "extern_bcc/edge_cleanup.cuh"
#include "extern_bcc/gpu_compress.cuh"

// n_s as it stood before this round's de-merger minted any leaver id.
int g_bcc_n_s_prev = 0;

void grow_temp_storage(GPU_BCG& g_bcg_ds, size_t needed) {
    if (needed <= g_bcg_ds.temp_storage_bytes) return;
    if (g_bcg_ds.d_temp_storage != NULL) {
        CUDA_CHECK(cudaFree(g_bcg_ds.d_temp_storage), "failed to free CUB temp storage");
    }
    CUDA_CHECK(cudaMalloc(&g_bcg_ds.d_temp_storage, needed), "failed to allocate CUB temp storage");
    g_bcg_ds.temp_storage_bytes = needed;
}

void excl_scan(const int* d_in, int* d_out, int n, GPU_BCG& g_bcg_ds) {
    if (n <= 0) return;

    size_t needed = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, needed, d_in, d_out, n),
               "excl_scan: failed to query temp storage size");
    grow_temp_storage(g_bcg_ds, needed);

    // CUB takes this by reference; keep the member's capacity intact.
    size_t bytes = g_bcg_ds.temp_storage_bytes;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(g_bcg_ds.d_temp_storage, bytes, d_in, d_out, n),
               "excl_scan: scan failed");
}

__device__ __forceinline__
uint64_t pack_edge(int a, int b) {
    int lo = min(a, b), hi = max(a, b);
    // Normalised before packing: remove_self_loops_duplicates dedups by sorted
    // u64 equality, so (a,b) and (b,a) would otherwise both survive.
    return (static_cast<uint64_t>(lo) << 32) | (static_cast<uint32_t>(hi));
}

// Counts the edges each batch edge will emit, and records which vertices must
// be pulled out of a block. Both stores to d_mark are the same value from
// every writer, so the race is benign.
__global__
void demerge_count_kernel(
    const uint64_t* d_batch, long batch_count,
    const int* d_mapping, int n,
    int* d_cnt, int* d_mark) {

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    uint64_t e = d_batch[idx];
    int u = e >> 32;
    int v = e & 0xFFFFFFFF;

    int mu = d_mapping[u];
    int mv = d_mapping[v];

    // Both endpoints already in one supernode: the edge can never create a new
    // articulation point, so it is dropped.
    if (mu == mv) {
        d_cnt[idx] = 0;
        return;
    }

    int extract_u = (mu >= n);
    int extract_v = (mv >= n);

    if (extract_u) d_mark[u] = 1;
    if (extract_v) d_mark[v] = 1;

    // Two edges per extraction: the leaver joins BOTH halves of its block.
    d_cnt[idx] = 1 + 2 * (extract_u + extract_v);
}

// Writes each batch edge's output block at its scanned offset.
__global__
void demerge_scatter_kernel(
    const uint64_t* d_batch, long batch_count,
    const int* d_mapping, int n,
    const int* d_off, uint64_t* d_out, long base) {

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    uint64_t e = d_batch[idx];
    int u = e >> 32;
    int v = e & 0xFFFFFFFF;

    int mu = d_mapping[u];
    int mv = d_mapping[v];
    if (mu == mv) return;

    long w = base + d_off[idx];

    // A leaver joins both halves of the block it came from, so it stays
    // 2-connected to the remnant (x-P and x-S, with the P-S edge closing the
    // triangle) -- and any two leavers of one block stay 2-connected to each
    // other through P and S. A single edge to the block hub would be a bridge
    // and would lose both. The mapping holds the PRIMARY id, so the other half
    // is simply the next one.
    if (mu >= n) { d_out[w++] = pack_edge(u, mu); d_out[w++] = pack_edge(u, mu + 1); }
    if (mv >= n) { d_out[w++] = pack_edge(v, mv); d_out[w++] = pack_edge(v, mv + 1); }

    // The batch edge itself. Both endpoints are standalone once the
    // extractions below are applied, so this is (u,v) verbatim.
    d_out[w] = pack_edge(u, v);
}

// Separate launch, after the scatter: the scatter must read the pre-extraction
// mapping. Driven off d_mark rather than off the edge list so no thread can
// observe a half-updated mapping.
__global__
void apply_extraction_kernel(const int* d_mark, int* d_mapping, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (d_mark[idx]) d_mapping[idx] = idx;
}

long demerge(GPU_BCG& g_bcg_ds, long BatchSize, uint64_t* d_out, long base) {

    if (BatchSize <= 0) return 0;

    const int n = g_bcg_ds.orig_numVert;
    const long threadsPerBlock = maxThreadsPerBlock;
    const size_t edge_blocks = (BatchSize + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaMemset(g_bcg_ds.d_mark, 0, sizeof(int) * n), "demerge: failed to clear d_mark");

    demerge_count_kernel<<<edge_blocks, threadsPerBlock>>>(
        g_bcg_ds.d_edgelist, BatchSize,
        g_bcg_ds.d_mapping, n,
        g_bcg_ds.d_edge_out_count, g_bcg_ds.d_mark);
    CUDA_CHECK(cudaDeviceSynchronize(), "demerge: failed to synchronize count kernel");

    excl_scan(g_bcg_ds.d_edge_out_count, g_bcg_ds.d_edge_out_offset, (int)BatchSize, g_bcg_ds);

    // total = offset of the last edge + how many that edge emits
    int last_off = 0, last_cnt = 0;
    CUDA_CHECK(cudaMemcpy(&last_off, g_bcg_ds.d_edge_out_offset + (BatchSize - 1), sizeof(int), cudaMemcpyDeviceToHost),
               "demerge: failed to read last offset");
    CUDA_CHECK(cudaMemcpy(&last_cnt, g_bcg_ds.d_edge_out_count + (BatchSize - 1), sizeof(int), cudaMemcpyDeviceToHost),
               "demerge: failed to read last count");
    const long total = (long)last_off + (long)last_cnt;

    assert(base + total <= g_bcg_ds.max_allot);

    demerge_scatter_kernel<<<edge_blocks, threadsPerBlock>>>(
        g_bcg_ds.d_edgelist, BatchSize,
        g_bcg_ds.d_mapping, n,
        g_bcg_ds.d_edge_out_offset, d_out, base);
    CUDA_CHECK(cudaDeviceSynchronize(), "demerge: failed to synchronize scatter kernel");

    // Separate launch, last: everything above had to read the pre-batch mapping.
    const size_t vert_blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    apply_extraction_kernel<<<vert_blocks, threadsPerBlock>>>(g_bcg_ds.d_mark, g_bcg_ds.d_mapping, n);
    CUDA_CHECK(cudaDeviceSynchronize(), "demerge: failed to synchronize extraction kernel");

    if (g_verbose) {
        std::cout << "\t[single-pass] de-merger emitted " << total
                  << " edges from " << BatchSize << " batch edges\n";
    }

    return total;
}

/* ------------------------------ Phase 5 ------------------------------ */

// M[v] = I[M[v]] for every original vertex.
__global__
void compose_mapping_kernel(int* d_mapping, const int* I, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_mapping[idx] = I[d_mapping[idx]];
}

void single_pass_batch(GPU_BCG& g_bcg_ds, long BatchSize) {

    const int  n = g_bcg_ds.orig_numVert;
    const int  V = g_bcg_ds.V_max;
    const long threadsPerBlock = maxThreadsPerBlock;

    // ---- carry the active edge set into the scratch buffer -----------------
    if (g_bcg_ds.numEdges > 0) {
        CUDA_CHECK(cudaMemcpy(g_bcg_ds.d_edge_buffer, g_bcg_ds.updated_edgelist,
                              g_bcg_ds.numEdges * sizeof(uint64_t), cudaMemcpyDeviceToDevice),
                   "single_pass_batch: failed to carry active edges");
    }

    // ---- phase 3 -----------------------------------------------------------
    const long emitted = demerge(g_bcg_ds, BatchSize, g_bcg_ds.d_edge_buffer, g_bcg_ds.numEdges);
    long total = g_bcg_ds.numEdges + emitted;
    assert(total <= g_bcg_ds.max_allot);

    if (total == 0) { g_bcg_ds.numEdges = 0; return; }

    // ---- phase 4 (host) ----------------------------------------------------
    // Reused across batches so the buffers are not reallocated every round.
    static std::vector<uint64_t> h_edges, h_next;
    static std::vector<int>      h_I, h_map;
    static std::vector<char>     h_iscut, h_live;

    h_edges.resize(total);
    CUDA_CHECK(cudaMemcpy(h_edges.data(), g_bcg_ds.d_edge_buffer,
                          total * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "single_pass_batch: failed to read active edges");

    // Which node ids are currently live, so a new block does not reuse one.
    // Both sources are needed: a block that absorbed nobody appears only in the
    // edge list, and a block with no incident edge appears only in the mapping.
    h_map.resize(n);
    CUDA_CHECK(cudaMemcpy(h_map.data(), g_bcg_ds.d_mapping, n * sizeof(int), cudaMemcpyDeviceToHost),
               "single_pass_batch: failed to read mapping");
    h_live.assign(V, 0);
    for (int i = 0; i < n; ++i) h_live[h_map[i]] = 1;
    for (long i = 0; i < total; ++i) {
        h_live[(int)(h_edges[i] >> 32)] = 1;
        h_live[(int)(h_edges[i] & 0xFFFFFFFF)] = 1;
    }

    const int nblocks = serial_bcc(h_edges, n, V, h_live, h_I, h_iscut, &h_next);

    CUDA_CHECK(cudaMemcpy(g_bcg_ds.d_rep, h_I.data(), V * sizeof(int), cudaMemcpyHostToDevice),
               "single_pass_batch: failed to upload labels");

    // ---- phase 5 -----------------------------------------------------------
    size_t blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    compose_mapping_kernel<<<blocks, threadsPerBlock>>>(g_bcg_ds.d_mapping, g_bcg_ds.d_rep, n);
    CUDA_CHECK(cudaDeviceSynchronize(), "single_pass_batch: failed to compose mapping");

    // The next block-cut graph comes back from phase 4 already built, so there
    // is nothing to relabel -- only to upload and dedup. It is a forest, so it
    // always fits.
    long next_total = (long)h_next.size();
    assert(next_total <= g_bcg_ds.max_allot);

    if (next_total == 0) { g_bcg_ds.numEdges = 0; return; }

    CUDA_CHECK(cudaMemcpy(g_bcg_ds.d_edge_buffer, h_next.data(),
                          next_total * sizeof(uint64_t), cudaMemcpyHostToDevice),
               "single_pass_batch: failed to upload the next block-cut graph");

    remove_self_loops_duplicates(
        g_bcg_ds.d_edge_buffer,
        next_total,                     // updated in place to the surviving count
        g_bcg_ds.d_flags,
        g_bcg_ds.h_num_selected_out,
        g_bcg_ds.d_num_selected_out,
        g_bcg_ds.updated_edgelist);

    g_bcg_ds.numEdges = next_total;
    total = next_total;

    if (g_verbose) {
        std::cout << "\t[single-pass] blocks = " << nblocks
                  << ", active edges = " << total << "\n";
    }
}

/*====================================================================
 * The DENSE-ID path: Algorithm 3 with Compress-Graph on the GPU.
 *
 * Differs from the split-range path above in exactly the places dense ids
 * force it to. Ids are no longer stable, so:
 *
 *   - "x sits inside a block" is M[x] < n_b, not M[x] >= n. A singleton node
 *     routinely lands on an id != x under dense numbering, so the split-range
 *     test would misfire.
 *   - de-merging x can no longer be M[x] = x: slot x was renumbered away
 *     batches ago and x is not a node of H_{i-1} at all. A fresh id has to be
 *     MINTED above the current top, which is why the paper's Phases 1 and 2 --
 *     absent from the split-range path -- come back here.
 *   - batch edges name original vertices, but emitted edges must name CURRENT
 *     nodes, so both endpoints go through M (or through the mint).
 *==================================================================*/

// Phase 1 + the counting half of Phase 3, in one pass.
__global__
void dense_count_kernel(
    const uint64_t* d_batch, long batch_count,
    const int* d_mapping, int n_b_cur,
    int* d_cnt, int* d_mark) {

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    uint64_t e = d_batch[idx];
    int u = e >> 32;
    int v = e & 0xFFFFFFFF;

    int mu = d_mapping[u];
    int mv = d_mapping[v];

    // Internal to one block: cannot create an articulation point, so dropped.
    if (mu == mv) { d_cnt[idx] = 0; return; }

    int extract_u = (mu < n_b_cur);
    int extract_v = (mv < n_b_cur);

    if (extract_u) d_mark[u] = 1;
    if (extract_v) d_mark[v] = 1;

    d_cnt[idx] = 1 + (extract_u + extract_v);   // ONE link per leaver
}

// Phase 2: one fresh id per extracted vertex, dense from n_s upward.
// Exclusive, not inclusive: the inclusive form issues the first id as n_s+1 and
// never issues n_s itself.
__global__
void dense_mint_kernel(const int* d_mark, const int* d_scan, int* d_L, int n_s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && d_mark[idx]) d_L[idx] = n_s + d_scan[idx];
}

// Phase 3: emit. A leaver joins BOTH halves of the block it left.
__global__
void dense_scatter_kernel(
    const uint64_t* d_batch, long batch_count,
    const int* d_mapping, const int* d_mark, const int* d_L, int n_b_cur,
    const int* d_off, uint64_t* d_out, long base) {

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_count) return;

    uint64_t e = d_batch[idx];
    int u = e >> 32;
    int v = e & 0xFFFFFFFF;

    int mu = d_mapping[u];
    int mv = d_mapping[v];
    if (mu == mv) return;

    long w = base + d_off[idx];

    // M holds the PRIMARY half of a block, so the secondary is the next id.
    // One edge to the block would be a bridge: it would make the block a false
    // articulation point and strand both the leaver and the remnant.
    // ONE link per leaver: a block is a single node, and the filter in
    // Compress-Graph is what re-establishes 2-connectivity through it.
    if (mu < n_b_cur) d_out[w++] = pack_edge(d_L[u], mu);
    if (mv < n_b_cur) d_out[w++] = pack_edge(d_L[v], mv);

    // The batch edge itself, in CURRENT node ids.
    int hu = d_mark[u] ? d_L[u] : mu;
    int hv = d_mark[v] ? d_L[v] : mv;
    d_out[w] = pack_edge(hu, hv);
}

// Separate launch, after the scatter, and for EVERY marked x -- not only inside
// the merged branch. Doing it in the emit loop races the reads of M there, and
// doing it only for merged endpoints leaves an emitted edge referencing an L
// that was never written.
__global__
void dense_apply_kernel(const int* d_mark, const int* d_L, int* d_mapping, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && d_mark[idx]) d_mapping[idx] = d_L[idx];
}

long demerge_dense(GPU_BCG& g, long BatchSize, uint64_t* d_out, long base) {

    if (BatchSize <= 0) return 0;

    const int  n = g.orig_numVert;
    const long threads = maxThreadsPerBlock;
    const size_t eb = (BatchSize + threads - 1) / threads;
    const size_t vb = (n + threads - 1) / threads;
    const int n_b_cur = g.n_b;

    // Published for Fast_BCC's fence-keep mode: leaver ids are minted from
    // here upward, so ">= g_bcc_n_s_prev" identifies a vertex pulled out of a
    // block THIS round. See union_find_gpu_keepfence.
    g_bcc_n_s_prev = g.n_s;

    CUDA_CHECK(cudaMemset(g.d_mark, 0, sizeof(int) * n), "demerge_dense: clear d_mark");

    dense_count_kernel<<<eb, threads>>>(
        g.d_edgelist, BatchSize, g.d_mapping, n_b_cur, g.d_edge_out_count, g.d_mark);
    CUDA_CHECK(cudaDeviceSynchronize(), "demerge_dense: count");

    // Phase 2
    excl_scan(g.d_mark, g.d_mark_scan, n, g);
    CUDA_CHECK(cudaDeviceSynchronize(), "demerge_dense: mint scan");
    int last_m = 0, last_ms = 0;
    CUDA_CHECK(cudaMemcpy(&last_m,  g.d_mark      + (n - 1), sizeof(int), cudaMemcpyDeviceToHost), "demerge_dense: mark");
    CUDA_CHECK(cudaMemcpy(&last_ms, g.d_mark_scan + (n - 1), sizeof(int), cudaMemcpyDeviceToHost), "demerge_dense: scan");
    const int extracted = last_m + last_ms;

    dense_mint_kernel<<<vb, threads>>>(g.d_mark, g.d_mark_scan, g.d_L, g.n_s, n);
    CUDA_CHECK(cudaDeviceSynchronize(), "demerge_dense: mint");

    // n_t = n_s + #extracted, bounded by n_s + 2|B|
    g.n_s += extracted;
    assert(g.n_s + 1 <= g.V_fb && "V_fb too small for the transient graph");

    // Phase 3
    excl_scan(g.d_edge_out_count, g.d_edge_out_offset, (int)BatchSize, g);
    CUDA_CHECK(cudaDeviceSynchronize(), "demerge_dense: emit scan");
    int last_off = 0, last_cnt = 0;
    CUDA_CHECK(cudaMemcpy(&last_off, g.d_edge_out_offset + (BatchSize - 1), sizeof(int), cudaMemcpyDeviceToHost), "demerge_dense: off");
    CUDA_CHECK(cudaMemcpy(&last_cnt, g.d_edge_out_count  + (BatchSize - 1), sizeof(int), cudaMemcpyDeviceToHost), "demerge_dense: cnt");
    const long total = (long)last_off + (long)last_cnt;
    assert(base + total <= g.max_allot);

    dense_scatter_kernel<<<eb, threads>>>(
        g.d_edgelist, BatchSize, g.d_mapping, g.d_mark, g.d_L, n_b_cur,
        g.d_edge_out_offset, d_out, base);
    CUDA_CHECK(cudaDeviceSynchronize(), "demerge_dense: scatter");

    dense_apply_kernel<<<vb, threads>>>(g.d_mark, g.d_L, g.d_mapping, n);
    CUDA_CHECK(cudaDeviceSynchronize(), "demerge_dense: apply");

    if (g_verbose) {
        std::cout << "\t[dense] extracted " << extracted << ", n_t = " << g.n_s
                  << ", emitted " << total << " edges\n";
    }
    return total;
}

void single_pass_batch_gpu(GPU_BCG& g, long BatchSize) {

    const int  n = g.orig_numVert;
    const long threads = maxThreadsPerBlock;

    // Carry the active edge set forward; the de-merger appends to it.
    if (g.numEdges > 0) {
        CUDA_CHECK(cudaMemcpy(g.d_edge_buffer, g.updated_edgelist,
                              g.numEdges * sizeof(uint64_t), cudaMemcpyDeviceToDevice),
                   "single_pass_batch_gpu: carry active edges");
    }

    const long emitted = demerge_dense(g, BatchSize, g.d_edge_buffer, g.numEdges);
    long total = g.numEdges + emitted;
    assert(total <= g.max_allot);
    if (total == 0) { g.numEdges = 0; return; }

    // Compress-Graph reads from updated_edgelist and leaves the rebuilt graph
    // there, so hand it the buffer we just appended to.
    // Dedup BEFORE Compress-Graph, not only after it. The de-merger appends to
    // the carried edge set, and the edges it emits can repeat one already there
    // -- two batch edges pulling the same vertex out of one block both emit
    // (L[x], p) and (L[x], p+1). The host Tarjan tolerates that (it skips the
    // parent by EDGE ID, so a parallel pair reads as the 2-edge block it really
    // is), but Fast_BCC assumes a SIMPLE graph: its fence test f_v <= low_u is
    // satisfied by a back edge that lands exactly on the parent, so a duplicated
    // tree edge reads as a bridge and the block is never compressed.
    //
    // This also stages the list into updated_edgelist, which is where
    // Compress-Graph reads from, so it replaces a memcpy rather than adding a pass.
    remove_self_loops_duplicates(
        g.d_edge_buffer, total, g.d_flags,
        g.h_num_selected_out, g.d_num_selected_out, g.updated_edgelist);

    static int s_batch = 0;
    if (getenv("BCC_CHECK")) { bcc_labels_gpu(g, g.n_s, total); compress_check(g, g.n_s, total, s_batch); }
    ++s_batch;

    CompressResult r = compress_graph_gpu(g, g.n_s, total);

    // Phase 5: M[v] = I[M[v]]
    size_t vb = (n + threads - 1) / threads;
    compose_mapping_kernel<<<vb, threads>>>(g.d_mapping, g.d_newId, n);
    CUDA_CHECK(cudaDeviceSynchronize(), "single_pass_batch_gpu: compose mapping");

    g.n_s = r.n_s;
    g.n_b = r.n_b;

    // The rebuild emits a block's attachment edge once per internal edge that
    // touches it, so duplicates are expected and this pass is load-bearing.
    if (r.m_s == 0) { g.numEdges = 0; return; }

    long next_total = r.m_s;   // already sitting in d_edge_buffer
    remove_self_loops_duplicates(
        g.d_edge_buffer, next_total, g.d_flags,
        g.h_num_selected_out, g.d_num_selected_out, g.updated_edgelist);

    g.numEdges = next_total;

    if (g_verbose) {
        std::cout << "\t[dense] n_s = " << g.n_s << ", n_b = " << g.n_b
                  << ", active edges = " << g.numEdges << "\n";
    }
}
