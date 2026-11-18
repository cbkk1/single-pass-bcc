//---------------------------------------------------------------------
// Compress-Graph on the GPU. Contract and reasoning in gpu_compress.cuh.
//---------------------------------------------------------------------

#include <cassert>
#include <iostream>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "utility/cuda_utility.cuh"
#include "extern_bcc/gpu_compress.cuh"
#include "extern_bcc/single_pass.cuh"
#include "fast_bcc/fast_bcc.cuh"
#include "extern_bcc/serial_bcc.cuh"
#include <cstdlib>
#include <map>
#include <vector>
#include <set>

// defined in fast_bcc/connected_components.cu
int bcc_fence_keep_mode();
int bcc_fence_mode_resolved();

#define TB 1024

// set by the driver before the LAST batch; only read under BCC_FILTER_LAST.
bool g_bcc_last_batch = false;

__device__ __forceinline__
uint64_t pack(int a, int b) {
    int lo = min(a, b), hi = max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint32_t>(hi);
}

//---------------------------------------------------------------------
// Step A -- find the components, so we know what to hang off the super-root.
//
// Same lock-free union-find as the rest of the repo: links always point from
// the larger id to the smaller, so a class representative is its minimum, and
// rep[u] == u identifies one node per component.
//---------------------------------------------------------------------

__device__ __forceinline__
int uf_find(int i, int* lbl) {
    int j = i;
    if (lbl[j] == j) return j;
    do { j = lbl[j]; } while (lbl[j] != j);
    int tmp;
    while ((tmp = lbl[i]) > j) { lbl[i] = j; i = tmp; }
    return j;
}

__device__ __forceinline__
void uf_union(int src, int dst, int* lbl) {
    while (true) {
        int u = uf_find(src, lbl);
        int v = uf_find(dst, lbl);
        if (u == v) break;
        if (v > u) { int t = u; u = v; v = t; }
        if (u == atomicCAS(&lbl[u], u, v)) return;
    }
}

__global__
void uf_init_kernel(int* lbl, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lbl[i] = i;
}

__global__
void uf_edges_kernel(const uint64_t* d_edges, long m, int* lbl) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t e = d_edges[i];
    uf_union(static_cast<int>(e >> 32), static_cast<int>(e & 0xFFFFFFFF), lbl);
}

// Flag one representative per component. Compaction is left to the scan.
__global__
void uf_flag_roots_kernel(int* lbl, int* d_flag, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_flag[i] = (uf_find(i, lbl) == i) ? 1 : 0;
}

// Append (super_root, rep) for every component representative.
__global__
void uf_attach_kernel(const int* d_flag, const int* d_off,
                      uint64_t* d_out, long base, int super_root, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_flag[i]) d_out[base + d_off[i]] = pack(super_root, i);
}

//---------------------------------------------------------------------
// Step A' -- forest metadata for the Euler tour.
//
// A spanning forest of c trees yields c Euler circuits whose list rankings each
// restart at zero, so their positions collide. These give each component a
// disjoint, contiguous slice of the position space.
//
// A component of n_i vertices owns 2*n_i positions: 2*(n_i - 1) arcs, plus the
// two its root occupies (init_arrays seeds the root at both ends of its block).
// An ISOLATED vertex therefore still gets a 2-wide block and a valid degenerate
// interval, so it needs no special case anywhere downstream.
//
// Everything here is DETERMINISTIC -- it depends only on connectivity, never on
// which spanning forest the union-find race happens to produce -- so it can be
// computed independently of Fast_BCC's own forest and still describe it.
//---------------------------------------------------------------------

// A component's representative is its minimum id, so rep[v] == v marks one
// node per component, and that node is also the tree's root.
__global__
void forest_flag_roots_kernel(int* lbl, int* croot, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) croot[i] = (uf_find(i, lbl) == i) ? 1 : 0;
}

// cid[v]: dense component index. roots[j]: the root of component j.
__global__
void forest_cid_kernel(int* lbl, const int* croot, const int* rank,
                       int* cid, int* roots, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cid[i] = rank[uf_find(i, lbl)];
    if (croot[i]) roots[rank[i]] = i;
}

__global__
void forest_size_kernel(const int* cid, int* csize, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&csize[cid[i]], 1);
}

__global__
void forest_clear_kernel(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0;
}

// Width of component i, in arcs and in positions.
__global__
void forest_width_kernel(const int* csize, int* width, int c, bool positions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c) width[i] = positions ? 2 * csize[i] : 2 * (csize[i] - 1);
}

// Which nodes carry a parent edge. Scanning this gives each one a unique arc
// slot, replacing the closed form `thid - (thid > root)` that only compacts
// around a single root.
__global__
void forest_notroot_kernel(const int* croot, int* has, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) has[i] = croot[i] ? 0 : 1;
}

//---------------------------------------------------------------------
// Step C -- per-edge block labels.
//
// beta(x,y) = rep[the endpoint deeper in the Euler tour]. See the header for
// why one rule covers tree, back and cross edges alike.
//---------------------------------------------------------------------

__global__
void beta_kernel(const uint64_t* d_edges, long m,
                 const int* d_rep, const int* d_first_occ, int* d_beta) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t e = d_edges[i];
    int x = static_cast<int>(e >> 32);
    int y = static_cast<int>(e & 0xFFFFFFFF);
    d_beta[i] = d_first_occ[x] > d_first_occ[y] ? d_rep[x] : d_rep[y];
}

//---------------------------------------------------------------------
// Step D -- everything else follows from beta.
//---------------------------------------------------------------------

__global__
void init_derive_kernel(int* d_blk_edges, int* d_blk_struct,
                        int* d_first_beta, int* d_cut, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { d_blk_edges[i] = 0; d_blk_struct[i] = 0; d_first_beta[i] = -1; d_cut[i] = 0; }
}

// One pass: count the edges of each block, note whether it holds a carried
// block's structural pair edge, and let each node claim the first block label
// it sees.
//
// Ids below n_b_prev are the carried BLOCK nodes; everything else kept its own
// of blocks carried in from the last round, and no REAL edge can have both
// endpoints down there: an original edge's endpoints are de-merge mints (all
// >= n_s_prev) and a carried bridge joins two own-nodes (all >= 2*n_b_prev).
// So (x, x+1) with x even and x+1 < 2*n_b_prev is exactly a pair edge.
__global__
void claim_kernel(const uint64_t* d_edges, long m, const int* d_beta,
                  int n_b_prev, int* d_blk_edges, int* d_blk_struct,
                  int* d_first_beta) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t e = d_edges[i];
    int x = static_cast<int>(e >> 32);
    int y = static_cast<int>(e & 0xFFFFFFFF);
    int b = d_beta[i];
    atomicAdd(&d_blk_edges[b], 1);
    // A group holding a carried block node must compress no matter how few
    // edges it has: the block carries residents, so its id has to survive. A
    // block with a single attachment has a one-edge group, which the >= 2 test
    // would reject -- turning the block into an own-node and stranding every
    // resident that points at it. (The pair edge used to mask this by giving
    // every block a second edge of its own.)
    if (x < n_b_prev || y < n_b_prev) d_blk_struct[b] = 1;
    atomicCAS(&d_first_beta[x], -1, b);
    atomicCAS(&d_first_beta[y], -1, b);
}

// Second pass: a node incident to a block other than the one it claimed sits in
// two blocks at once, which is the definition of a cut vertex.
__global__
void cut_kernel(const uint64_t* d_edges, long m, const int* d_beta,
                const int* d_first_beta, int* d_cut) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t e = d_edges[i];
    int x = static_cast<int>(e >> 32);
    int y = static_cast<int>(e & 0xFFFFFFFF);
    int b = d_beta[i];
    if (d_first_beta[x] != b) d_cut[x] = 1;
    if (d_first_beta[y] != b) d_cut[y] = 1;
}

// A block compresses if it holds two or more edges, OR if its one edge is a
// carried block's pair edge. The second clause is load-bearing: a block that is
// a whole connected component arrives with no attachments and no new edges, so
// its pair edge is the component's ONLY edge. Reading that as a bridge splits
// the block into two plain nodes, destroys the pair, and -- because the merged
// test is M[x] < 2*n_b -- silently strands every resident it held, which then
// never gets de-merged again.
__global__
// min_edges is 2 by default: a BCC with ONE edge is a bridge, whose endpoints
// are not 2-connected, so collapsing them to one identity is a different
// representation, not a refinement. BCC_MIN_EDGES=1 switches to compressing
// every BCC, which is sound (a non-cut endpoint of a bridge has degree 1, so
// the resulting node has degree <= 1 and can fuse nothing) but changes the id
// budget and the edge count -- see the measurement in the commit message.
void blk_flag_kernel(const int* d_blk_edges, const int* d_blk_struct,
                     const int* d_first_beta,
                     int n_b_prev, int* d_blk_flag, int n, int min_edges) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // A carried block nothing touched this round has no edge in H at all, so it
    // appears in no beta class. It is a finished BCC and must keep a block id.
    int carried_alone = (i < n_b_prev && d_first_beta[i] < 0);
    d_blk_flag[i] = (d_blk_edges[i] >= min_edges || d_blk_struct[i] || carried_alone) ? 1 : 0;
}

// Read once; 2 unless BCC_MIN_EDGES says otherwise.
static int blk_min_edges() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("BCC_MIN_EDGES"); v = (e && atoi(e) == 1) ? 1 : 2; }
    return v;
}

//---------------------------------------------------------------------
// Step D' -- THE FILTER.
//
// A carried block node stands for a 2-connected set S. Every edge of H at that
// node came from a DISTINCT vertex of S: a vertex absorbed into S has all its
// edges inside S, so an endpoint that still reaches outside is a cut vertex and
// was never absorbed. S being 2-connected, any two such vertices are joined by
// two internally disjoint paths, so everything attached to the node lies in ONE
// biconnected component of G. Bcc cannot see that -- the block's internal edges
// were deleted by compression, so the node reads as an articulation point --
// hence this pass unions the blocks it reported.
//
// This is what makes a ONE-node block sound, and is why no pair edge is needed.
//---------------------------------------------------------------------

// |f^-1(x)| per H node, saturated at two -- only "is it exactly 1?" is asked.
// This is Euro-Par's freq_cal. Reuses d_counter, which our path never touches.
__global__
void fcnt_clear_kernel(int* cnt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cnt[i] = 0;
}
__global__
void fcnt_kernel(const int* d_mapping, int* cnt, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int x = d_mapping[v];
    if (x >= 0 && cnt[x] < 2) atomicAdd(&cnt[x], 1);
}

// Which nodes does the filter merge at?
//   mode 0 (id)  : x is a carried block node            -- our predicate
//   mode 1 (dis) : that, OR |f^-1(x)| != 1              -- id-range OR Euro-Par's
// They differ in two places: a block with exactly ONE resident (id catches it,
// |f^-1| calls it real) and an EMPTY own-node, which can only exist if an
// earlier round skipped the filter and a block got demoted (|f^-1| catches it,
// id calls it real).
//   mode 2 (cnt) : |f^-1(x)| != 1 ALONE -- exactly BCC_Repair's predicate,
//                  with no id clause, so it isolates the question "is the
//                  filter just repair moved earlier?".
__device__ __forceinline__
bool filt_false(int x, int n_b_prev, const int* cnt, int mode) {
    if (mode == 2) return cnt[x] != 1;
    if (x < n_b_prev) return true;
    return mode ? (cnt[x] != 1) : false;
}

__global__
void filter_union_kernel(const uint64_t* d_edges, long m, const int* d_beta,
                         const int* d_first_beta, int n_b_prev,
                         const int* cnt, int mode, int* par) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t e = d_edges[i];
    int x = static_cast<int>(e >> 32);
    int y = static_cast<int>(e & 0xFFFFFFFF);
    int b = d_beta[i];
    if (filt_false(x, n_b_prev, cnt, mode)) { int a = d_first_beta[x]; if (a >= 0) uf_union(b, a, par); }
    if (filt_false(y, n_b_prev, cnt, mode)) { int a = d_first_beta[y]; if (a >= 0) uf_union(b, a, par); }
}

__global__
void filter_apply_kernel(int* d_beta, long m, int* par) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) d_beta[i] = uf_find(d_beta[i], par);
}

// A node keeps its own identity unless it is absorbed, and it is absorbed only
// if it is non-cut AND its one block actually compresses. Isolated nodes
// (first_beta == -1) and bridge endpoints keep their identity too.
__global__
void own_kernel(const int* d_cut, const int* d_first_beta,
                const int* d_blk_flag, int n_b_prev, int* d_own, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = d_first_beta[i];
    if (d_cut[i])  { d_own[i] = 1; return; }
    if (b < 0)     { d_own[i] = (i < n_b_prev) ? 0 : 1; return; }  // carried alone
    d_own[i] = d_blk_flag[b] ? 0 : 1;
}

//---------------------------------------------------------------------
// Step E -- the dense renumbering.
//---------------------------------------------------------------------

__global__
void newid_kernel(const int* d_own, const int* d_first_beta,
                  const int* d_scan_b, const int* d_scan_o,
                  int n_b, int* d_newId, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (d_own[i]) { d_newId[i] = n_b + d_scan_o[i]; return; }   // keeps identity
    int b = d_first_beta[i];
    d_newId[i] = d_scan_b[(b < 0) ? i : b];   // its block's id (b<0: carried alone)
}

//---------------------------------------------------------------------
// Step F -- rebuild.
//
//   a one-edge block   -> carried over, both ends mapped through I
//   a compressed block -> the pair edge, plus both halves to each attachment
//   anything else      -> dropped (internal to a block, between two residents)
//
// Rebuilt rather than relabelled in place, because relabelling cannot express
// the last line: an edge between two cut vertices of one block maps to itself
// and survives looking exactly like a bridge.
//---------------------------------------------------------------------

__global__
void rebuild_count_kernel(const uint64_t* d_edges, long m, const int* d_beta,
                          const int* d_blk_flag, const int* d_own, int* d_cnt) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t e = d_edges[i];
    int x = static_cast<int>(e >> 32);
    int y = static_cast<int>(e & 0xFFFFFFFF);
    d_cnt[i] = d_blk_flag[d_beta[i]] ? (d_own[x] + d_own[y]) : 1;
}

__global__
void rebuild_scatter_kernel(const uint64_t* d_edges, long m, const int* d_beta,
                            const int* d_blk_flag, const int* d_own,
                            const int* d_newId, const int* d_scan_b,
                            const int* d_off, uint64_t* d_out) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t e = d_edges[i];
    int x = static_cast<int>(e >> 32);
    int y = static_cast<int>(e & 0xFFFFFFFF);
    int b = d_beta[i];
    long w = d_off[i];

    if (!d_blk_flag[b]) { d_out[w] = pack(d_newId[x], d_newId[y]); return; }

    int p = d_scan_b[b];
    if (d_own[x]) d_out[w++] = pack(d_newId[x], p);
    if (d_own[y]) d_out[w++] = pack(d_newId[y], p);
}

//---------------------------------------------------------------------

// Total of an exclusive scan of n items: scan[n-1] + in[n-1], both on device.
static int scan_total(const int* d_in, const int* d_scan, int n) {
    if (n <= 0) return 0;
    int last_in = 0, last_scan = 0;
    CUDA_CHECK(cudaMemcpy(&last_in,   d_in   + (n - 1), sizeof(int), cudaMemcpyDeviceToHost), "scan_total in");
    CUDA_CHECK(cudaMemcpy(&last_scan, d_scan + (n - 1), sizeof(int), cudaMemcpyDeviceToHost), "scan_total scan");
    return last_in + last_scan;
}

// Computes everything Fast_BCC's Euler tour needs in order to run on a FOREST
// instead of a single tree. See the Step A' comment above.
static void forest_metadata(GPU_BCG& g, int n_t) {
    const int threads = TB;

    uf_init_kernel<<<CEIL(n_t, threads), threads>>>(g.d_cc, n_t);
    if (g.numEdges > 0)
        uf_edges_kernel<<<CEIL(g.numEdges, threads), threads>>>(
            g.updated_edgelist, g.numEdges, g.d_cc);
    forest_flag_roots_kernel<<<CEIL(n_t, threads), threads>>>(g.d_cc, g.d_croot, n_t);
    CUDA_CHECK(cudaDeviceSynchronize(), "forest: union-find");

    // rank among roots == dense component index
    excl_scan(g.d_croot, g.d_csize, n_t, g);
    CUDA_CHECK(cudaDeviceSynchronize(), "forest: root ranks");
    const int c = scan_total(g.d_croot, g.d_csize, n_t);
    g.n_comp = c;

    forest_cid_kernel<<<CEIL(n_t, threads), threads>>>(
        g.d_cc, g.d_croot, g.d_csize, g.d_cid, g.d_roots, n_t);
    CUDA_CHECK(cudaDeviceSynchronize(), "forest: component ids");

    forest_clear_kernel<<<CEIL(c + 1, threads), threads>>>(g.d_csize, c + 1);
    CUDA_CHECK(cudaDeviceSynchronize(), "forest: clear sizes");
    forest_size_kernel<<<CEIL(n_t, threads), threads>>>(g.d_cid, g.d_csize, n_t);
    CUDA_CHECK(cudaDeviceSynchronize(), "forest: sizes");

    // Two offset tables. compute_level works in ARC index space [0, E), while
    // compute_first_last_occ works in POSITION space [0, 2N) -- hence abase and
    // pbase, which differ by 2 per component.
    forest_width_kernel<<<CEIL(c, threads), threads>>>(g.d_csize, g.d_blk_edges, c, false);
    CUDA_CHECK(cudaDeviceSynchronize(), "forest: arc widths");
    excl_scan(g.d_blk_edges, g.d_abase, c, g);

    forest_width_kernel<<<CEIL(c, threads), threads>>>(g.d_csize, g.d_blk_edges, c, true);
    CUDA_CHECK(cudaDeviceSynchronize(), "forest: position widths");
    excl_scan(g.d_blk_edges, g.d_pbase, c, g);

    // Arc slot per node that has a parent edge. A node has one exactly when it
    // is NOT its component's root, so this needs no parent-edge array -- which
    // matters, because Fast_BCC's union-find has not run yet.
    forest_notroot_kernel<<<CEIL(n_t, threads), threads>>>(g.d_croot, g.d_blk_flag, n_t);
    CUDA_CHECK(cudaDeviceSynchronize(), "forest: non-roots");
    excl_scan(g.d_blk_flag, g.d_slot, n_t, g);
    CUDA_CHECK(cudaDeviceSynchronize(), "forest: arc slots");
}

void bcc_labels_gpu(GPU_BCG& g, int n_t, long m_a) {

    const int threads = TB;
    assert(n_t <= g.V_fb && "V_fb too small for the transient graph");

    // ---- A. Fast_BCC, steps 1-3, on the FOREST ---------------------------
    const int  save_numVert = g.numVert;
    const long save_numEdges = g.numEdges;
    g.numVert  = n_t;
    g.numEdges = m_a;
    forest_metadata(g, n_t);
    Fast_BCC_labels(g);
    g.numVert  = save_numVert;
    g.numEdges = save_numEdges;

    const int n_aug = n_t;   // no augmentation: the tour runs on the forest itself

    if (getenv("BCC_SLOT")) {
        std::vector<int> cr(n_t), sl(n_t);
        std::vector<uint64_t> pe(n_t);
        CUDA_CHECK(cudaMemcpy(cr.data(), g.d_croot,      n_t*4, cudaMemcpyDeviceToHost), "cr");
        CUDA_CHECK(cudaMemcpy(sl.data(), g.d_slot,       n_t*4, cudaMemcpyDeviceToHost), "sl");
        CUDA_CHECK(cudaMemcpy(pe.data(), g.d_parentEdge, n_t*8, cudaMemcpyDeviceToHost), "pe");
        int nroot = 0, nedge = 0, disagree = 0;
        std::vector<int> used(n_t + 1, 0);
        int dup = 0, oob = 0;
        for (int i = 0; i < n_t; ++i) {
            if (cr[i]) ++nroot; else {
                ++nedge;
                if (sl[i] < 0 || sl[i] > n_t) ++oob; else if (used[sl[i]]++) ++dup;
            }
            bool fb_root = (pe[i] == (uint64_t)INT_MAX);
            if (fb_root != (cr[i] != 0)) ++disagree;
        }
        const int E2 = n_t - g.n_comp;
        std::cout << "  [slot] n_t=" << n_t << " c=" << g.n_comp
                  << " my_roots=" << nroot << " nonroots=" << nedge
                  << " E/2=" << E2
                  << "  slot_dup=" << dup << " slot_oob=" << oob
                  << "  CROOT_vs_FASTBCC_disagree=" << disagree << "\n";
    }
    if (getenv("BCC_OCC")) {
        std::vector<int> fo(n_t), lo(n_t), lv(n_t), cr(n_t), pa(n_t), cid(n_t);
        CUDA_CHECK(cudaMemcpy(fo.data(), g.d_first_occ, n_t*4, cudaMemcpyDeviceToHost), "fo");
        CUDA_CHECK(cudaMemcpy(lo.data(), g.d_last_occ,  n_t*4, cudaMemcpyDeviceToHost), "lo");
        CUDA_CHECK(cudaMemcpy(lv.data(), g.d_level,     n_t*4, cudaMemcpyDeviceToHost), "lv");
        CUDA_CHECK(cudaMemcpy(cr.data(), g.d_croot,     n_t*4, cudaMemcpyDeviceToHost), "cr");
        CUDA_CHECK(cudaMemcpy(pa.data(), g.d_parent,    n_t*4, cudaMemcpyDeviceToHost), "pa");
        CUDA_CHECK(cudaMemcpy(cid.data(), g.d_cid,      n_t*4, cudaMemcpyDeviceToHost), "cid");
        int bad = 0;
        for (int i = 0; i < n_t && bad < 5; ++i) {
            if (fo[i] >= 0 && lo[i] >= 0) continue;
            int p = pa[i];
            std::vector<int> cs(g.n_comp);
            CUDA_CHECK(cudaMemcpy(cs.data(), g.d_csize, g.n_comp*4, cudaMemcpyDeviceToHost), "cs");
            std::cout << "  [bad] node " << i << " root=" << (int)cr[i]
                      << " cid=" << cid[i] << " compsize=" << cs[cid[i]]
                      << " first=" << fo[i] << " last=" << lo[i]
                      << " level=" << lv[i]
                      << " parent=" << p << " parent_level=" << (p>=0&&p<n_t?lv[p]:-999)
                      << " parent_is_self=" << (p==i) << "\n";
            ++bad;
        }
        if (bad) std::cout << "  [bad] n_t=" << n_t << " c=" << g.n_comp << "  total flagged>=" << bad << "\n";
    }

    // ---- C. per-edge block labels, over the ORIGINAL edges only ----------
    if (m_a > 0)
        beta_kernel<<<CEIL(m_a, threads), threads>>>(
            g.updated_edgelist, m_a, g.d_rep, g.d_first_occ, g.d_beta);

    // ---- D. block sizes, cut status, own status --------------------------
    // g.n_b is still LAST round's block count here -- it is only advanced after
    // compress returns -- so ids below it are exactly the carried BLOCK nodes.
    const int n_b_prev = g.n_b;

    // BCC_BLKEDGE=1: classify every edge of H incident to a CARRIED BLOCK node
    // into "fresh de-merge link" (other end >= n_s_prev, minted by this round's
    // de-merger) vs "carried" (anything else -- an attachment from an earlier
    // round, or a cycle-edge remnant left by compression). The filter has to
    // union BOTH kinds; a rule that keeps only the freshly added de-merge edges
    // sees only the first column.
    if (getenv("BCC_BLKEDGE") && m_a > 0 && n_b_prev > 0) {
        extern int g_bcc_n_s_prev;
        std::vector<uint64_t> he(m_a);
        CUDA_CHECK(cudaMemcpy(he.data(), g.updated_edgelist, m_a*8, cudaMemcpyDeviceToHost), "blkedge");
        int fresh = 0, carried = 0;
        for (uint64_t e : he) {
            int x = (int)(e >> 32), y = (int)(e & 0xFFFFFFFF);
            bool bx = x < n_b_prev, by = y < n_b_prev;
            if (!bx && !by) continue;
            int other = bx ? y : x;
            if (bx && by) { ++carried; continue; }
            if (other >= g_bcc_n_s_prev) ++fresh; else ++carried;
        }
        printf("        [blkedge] at carried blocks: fresh de-merge links=%d  carried edges=%d\n",
               fresh, carried);
    }

    init_derive_kernel<<<CEIL(n_aug, threads), threads>>>(
        g.d_blk_edges, g.d_blk_struct, g.d_first_beta, g.iscutVertex, n_aug);
    CUDA_CHECK(cudaDeviceSynchronize(), "init derive");

    if (m_a > 0) {
        // Pass 1 only needs d_first_beta, as the anchor the filter unions onto.
        claim_kernel<<<CEIL(m_a, threads), threads>>>(
            g.updated_edgelist, m_a, g.d_beta, n_b_prev,
            g.d_blk_edges, g.d_blk_struct, g.d_first_beta);
        CUDA_CHECK(cudaDeviceSynchronize(), "claim (anchors)");

        // ---- D'. the filter -------------------------------------------
        // BCC_FILTER_FROM=k : skip the filter on the first k rounds, so k =
        // (batches-1) means "filter only at the very end".
        static int s_round = 0;
        const char* ff = getenv("BCC_FILTER_FROM");
        // Step 4 of Fast_BCC now declines to fence a tree edge whose PARENT is
        // a carried block, which produces the merged classes directly -- so
        // this pass, and the re-derivation it forces below, are not needed.
        // It survives only for BCC_FENCE_KEEP=0, which restores the v2 path.
        bool do_filter = bcc_fence_mode_resolved() == 0
                      && !getenv("BCC_NOFILTER")
                      && (!ff || s_round >= atoi(ff))
                      && (!getenv("BCC_FILTER_LAST") || g_bcc_last_batch);
        ++s_round;
        // BCC_PRED=dis widens the realness test; default "id" is unchanged.
        static int s_mode = -1;
        if (s_mode < 0) { const char* pm = getenv("BCC_PRED");
                          s_mode = !pm ? 0 : (pm[0] == 'd' ? 1 : (pm[0] == 'c' ? 2 : 0)); }
        if (s_mode >= 1) {
            fcnt_clear_kernel<<<CEIL(n_aug, threads), threads>>>(g.d_counter, n_aug);
            CUDA_CHECK(cudaDeviceSynchronize(), "fcnt clear");
            fcnt_kernel<<<CEIL(g.orig_numVert, threads), threads>>>(
                g.d_mapping, g.d_counter, g.orig_numVert);
            CUDA_CHECK(cudaDeviceSynchronize(), "fcnt");
        }
        if ((n_b_prev > 0 || s_mode >= 1) && do_filter) {
            uf_init_kernel<<<CEIL(n_aug, threads), threads>>>(g.d_cc, n_aug);
            CUDA_CHECK(cudaDeviceSynchronize(), "filter init");
            filter_union_kernel<<<CEIL(m_a, threads), threads>>>(
                g.updated_edgelist, m_a, g.d_beta, g.d_first_beta, n_b_prev,
                g.d_counter, s_mode, g.d_cc);
            CUDA_CHECK(cudaDeviceSynchronize(), "filter union");
            filter_apply_kernel<<<CEIL(m_a, threads), threads>>>(g.d_beta, m_a, g.d_cc);
            CUDA_CHECK(cudaDeviceSynchronize(), "filter apply");

            // Everything derived from beta must be recomputed on the MERGED
            // labels -- edge counts, anchors and, above all, the cut set: a
            // node that looked like a cut vertex only because a block node
            // split its blocks is not one.
            init_derive_kernel<<<CEIL(n_aug, threads), threads>>>(
                g.d_blk_edges, g.d_blk_struct, g.d_first_beta, g.iscutVertex, n_aug);
            CUDA_CHECK(cudaDeviceSynchronize(), "re-init derive");
            claim_kernel<<<CEIL(m_a, threads), threads>>>(
                g.updated_edgelist, m_a, g.d_beta, n_b_prev,
                g.d_blk_edges, g.d_blk_struct, g.d_first_beta);
            CUDA_CHECK(cudaDeviceSynchronize(), "claim (merged)");
        }

        cut_kernel<<<CEIL(m_a, threads), threads>>>(
            g.updated_edgelist, m_a, g.d_beta, g.d_first_beta, g.iscutVertex);
        CUDA_CHECK(cudaDeviceSynchronize(), "cut");
    }

    if (getenv("BCC_FILT") && n_b_prev > 0 && m_a > 0) {
        // The filter's own invariant: having unioned every group at a carried
        // block node, that node must now lie in exactly ONE group, i.e. it
        // must not read as a cut vertex. If it still does, the merge missed.
        std::vector<int> cutv(n_aug), fb(n_aug);
        CUDA_CHECK(cudaMemcpy(cutv.data(), g.iscutVertex, n_aug*4, cudaMemcpyDeviceToHost), "c");
        CUDA_CHECK(cudaMemcpy(fb.data(), g.d_first_beta, n_aug*4, cudaMemcpyDeviceToHost), "f");
        int bad = 0, first = -1;
        for (int x = 0; x < n_b_prev && x < n_aug; ++x)
            if (cutv[x]) { if (first < 0) first = x; ++bad; }
        if (bad)
            std::cout << "  [FILT] n_t=" << n_t << " m_a=" << m_a
                      << " n_b_prev=" << n_b_prev
                      << "  block nodes STILL cut = " << bad
                      << " first=" << first << " first_beta=" << fb[first] << "\n";
    }

    blk_flag_kernel<<<CEIL(n_aug, threads), threads>>>(
        g.d_blk_edges, g.d_blk_struct, g.d_first_beta, n_b_prev,
        g.d_blk_flag, n_aug, blk_min_edges());
    CUDA_CHECK(cudaDeviceSynchronize(), "block flags");

    // BCC_BETA: show WHY each beta class was or was not made a block. This is
    // the decision the filter has to precede: d_blk_struct promotes a class to
    // a block merely for touching a carried block node, so two bridges hanging
    // off one carried block become two blocks unless their classes were merged
    // first.
    if (getenv("BCC_BETA") && m_a > 0) {
        std::vector<int> be(n_aug), bs(n_aug), bf(n_aug), bt(m_a);
        std::vector<uint64_t> he(m_a);
        CUDA_CHECK(cudaMemcpy(be.data(), g.d_blk_edges,  n_aug*4, cudaMemcpyDeviceToHost), "be");
        CUDA_CHECK(cudaMemcpy(bs.data(), g.d_blk_struct, n_aug*4, cudaMemcpyDeviceToHost), "bs");
        CUDA_CHECK(cudaMemcpy(bf.data(), g.d_blk_flag,   n_aug*4, cudaMemcpyDeviceToHost), "bf");
        CUDA_CHECK(cudaMemcpy(bt.data(), g.d_beta,       m_a*4,   cudaMemcpyDeviceToHost), "bt");
        CUDA_CHECK(cudaMemcpy(he.data(), g.updated_edgelist, m_a*8, cudaMemcpyDeviceToHost), "he");
        std::cout << "  [BETA] n_b_prev=" << n_b_prev << "  classes:\n";
        std::set<int> seen(bt.begin(), bt.end());
        for (int b : seen) {
            std::cout << "    beta " << b << " edges:";
            for (long e = 0; e < m_a; ++e) if (bt[e] == b)
                std::cout << " (" << (int)(he[e] >> 32) << ","
                          << (int)(he[e] & 0xFFFFFFFF) << ")";
            std::cout << "   |E|=" << be[b] << " struct=" << bs[b]
                      << " -> " << (bf[b] ? "BLOCK" : "no") << "\n";
        }
    }

    // Slot n_aug is NOT a block. compress_graph_gpu scans d_blk_flag over
    // n_t + 1 so scan_total can read one past the end, but nothing above ever
    // WRITES index n_aug. Left alone it keeps whatever the PREVIOUS batch put
    // there and inflates n_b by one.
    CUDA_CHECK(cudaMemset(g.d_blk_flag + n_aug, 0, sizeof(int)), "clear blk_flag sentinel");

    own_kernel<<<CEIL(n_t, threads), threads>>>(
        g.iscutVertex, g.d_first_beta, g.d_blk_flag, n_b_prev, g.d_own, n_t);
    CUDA_CHECK(cudaDeviceSynchronize(), "own flags");
}

CompressResult compress_graph_gpu(GPU_BCG& g, int n_t, long m_a) {

    const int threads = TB;

    bcc_labels_gpu(g, n_t, m_a);

    // ---- E. dense renumbering -------------------------------------------
    // Two scans, because a node's new id comes from a different counter
    // depending on whether it is absorbed. Blocks are ranked over the whole
    // label space (a label is a node id), own-nodes over [0, n_t).
    excl_scan(g.d_blk_flag, g.d_scan_b, n_t + 1, g);
    excl_scan(g.d_own,      g.d_scan_o, n_t,     g);
    CUDA_CHECK(cudaDeviceSynchronize(), "renumbering scans");

    const int n_b = scan_total(g.d_blk_flag, g.d_scan_b, n_t + 1);
    const int n_o = scan_total(g.d_own,      g.d_scan_o, n_t);
    const int n_s = n_b + n_o;
    assert(n_s <= g.V_fb && "V_fb too small for the rebuilt graph");

    newid_kernel<<<CEIL(n_t, threads), threads>>>(
        g.d_own, g.d_first_beta, g.d_scan_b, g.d_scan_o, n_b, g.d_newId, n_t);
    CUDA_CHECK(cudaDeviceSynchronize(), "new ids");

    // ---- F. rebuild ------------------------------------------------------
    long m_s = 0;
    if (m_a > 0) {
        rebuild_count_kernel<<<CEIL(m_a, threads), threads>>>(
            g.updated_edgelist, m_a, g.d_beta, g.d_blk_flag, g.d_own, g.d_cnt);
        CUDA_CHECK(cudaDeviceSynchronize(), "rebuild count");

        excl_scan(g.d_cnt, g.d_off, static_cast<int>(m_a), g);
        CUDA_CHECK(cudaDeviceSynchronize(), "rebuild scan");
        m_s = scan_total(g.d_cnt, g.d_off, static_cast<int>(m_a));

        assert(m_s + n_b <= g.edge_capacity && "edge_capacity too small for the rebuild");

        // Into d_edge_buffer, not in place: the scatter reads the edge it is
        // replacing, and a block's attachment edges land far from it.
        rebuild_scatter_kernel<<<CEIL(m_a, threads), threads>>>(
            g.updated_edgelist, m_a, g.d_beta, g.d_blk_flag, g.d_own,
            g.d_newId, g.d_scan_b, g.d_off, g.d_edge_buffer);
        CUDA_CHECK(cudaDeviceSynchronize(), "rebuild scatter");
    }

    // Left in d_edge_buffer deliberately -- the caller's next step is the
    // dedup, which needs a separate destination anyway. Copying it back to
    // updated_edgelist here would be two full-buffer memcpys for nothing.
    return CompressResult{n_s, n_b, m_s};
}

//---------------------------------------------------------------------
// Debug: compare one batch's Compress-Graph against the serial Tarjan on the
// SAME transient graph. Enabled with BCC_CHECK=1. Reports the first batch whose
// block partition or cut set diverges, which separates a Compress-Graph fault
// from a de-merger fault.
//---------------------------------------------------------------------
void compress_check(GPU_BCG& g, int n_t, long m_a, int batch) {
    if (!getenv("BCC_CHECK")) return;

    std::vector<uint64_t> h_e(m_a);
    CUDA_CHECK(cudaMemcpy(h_e.data(), g.updated_edgelist, m_a * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost), "check: edges");
    std::vector<int> h_beta(m_a), h_cut(n_t);
    CUDA_CHECK(cudaMemcpy(h_beta.data(), g.d_beta, m_a * sizeof(int),
                          cudaMemcpyDeviceToHost), "check: beta");
    CUDA_CHECK(cudaMemcpy(h_cut.data(), g.iscutVertex, n_t * sizeof(int),
                          cudaMemcpyDeviceToHost), "check: cut");

    std::vector<int>  r_beta;
    std::vector<char> r_cut;
    std::vector<long> r_sz;
    serial_bcc_partition(h_e.data(), m_a, n_t, r_beta, r_cut, r_sz);

    std::map<int,int> fwd, rev;
    long bad_part = 0;
    for (long e = 0; e < m_a; ++e) {
        auto f = fwd.emplace(h_beta[e], r_beta[e]);
        if (!f.second && f.first->second != r_beta[e]) ++bad_part;
        auto r = rev.emplace(r_beta[e], h_beta[e]);
        if (!r.second && r.first->second != h_beta[e]) ++bad_part;
    }
    long bad_cut = 0;
    for (int v = 0; v < n_t; ++v) if ((h_cut[v] != 0) != (r_cut[v] != 0)) ++bad_cut;

    if (bad_part || bad_cut)
        std::cout << "  [check] batch " << batch << ": n_t=" << n_t << " m_a=" << m_a
                  << "  partition mismatches=" << bad_part
                  << "  cut mismatches=" << bad_cut << "\n";
}
