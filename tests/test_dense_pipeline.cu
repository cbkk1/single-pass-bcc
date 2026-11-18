//---------------------------------------------------------------------
// End-to-end test of the dense-id single-pass pipeline (Algorithm 3) with
// Compress-Graph on the GPU.
//
// Streams a random graph through in a random number of batches and checks
// three things against the serial Tarjan over the whole edge set:
//
//   1. BCC COUNT. In the finished BCG a block owns an aligned pair in
//      [0, 2*n_b) and everything else owns an id above it, so a bridge is
//      exactly an edge with both endpoints >= 2*n_b -- every other edge is a
//      block's pair edge or an attachment to one. Hence
//          #BCC = n_b + #{ (x,y) in E : x >= 2*n_b and y >= 2*n_b }.
//   2. NO REAL BCC IS SPLIT. All non-cut members of a true BCC must end up
//      sharing one M value. This is the property the star attachment used to
//      break.
//   3. NO OVER-MERGE. Two vertices in no common BCC must not share an M value.
//
// Build:
//   nvcc -std=c++17 -O2 --expt-relaxed-constexpr -I include \
//        tests/test_dense_pipeline.cu src/extern_bcc/gpu_compress.cu \
//        src/extern_bcc/single_pass.cu src/extern_bcc/serial_bcc.cu \
//        src/extern_bcc/bcg_memory_utils.cu src/extern_bcc/edge_cleanup.cu \
//        src/fast_bcc/*.cu src/utility/cuda_utility.cu -o build/test_dense_pipeline
//---------------------------------------------------------------------

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>

#include <cuda_runtime.h>

#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"
#include "extern_bcc/gpu_compress.cuh"
#include "extern_bcc/single_pass.cuh"
#include "extern_bcc/serial_bcc.cuh"

extern bool g_bcc_last_batch;
bool g_verbose          = false;
int  local_block_size   = 1000;
long maxThreadsPerBlock = 0;

static uint64_t pk(int a, int b) {
    int lo = std::min(a, b), hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint32_t>(hi);
}

int main(int argc, char** argv) {
    int seed0  = (argc > 1) ? atoi(argv[1]) : 1;
    int trials = (argc > 2) ? atoi(argv[2]) : 200;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0), "failed to query device");
    maxThreadsPerBlock = prop.maxThreadsPerBlock;

    const int  N  = 128;
    const long BS = 256;

    int fail_count = 0, fail_split = 0, fail_merge = 0, checked = 0, fail_empty = 0, fail_false = 0, fail_thin = 0;

    for (int t = 0; t < trials; ++t) {
        std::mt19937 rng(seed0 + t);

        int n = 6 + rng() % (N - 6);
        std::set<uint64_t> uniq;
        long want = 1 + rng() % (3 * n);
        for (long i = 0; i < want; ++i) {
            int u = rng() % n, v = rng() % n;
            if (u != v) uniq.insert(pk(u, v));
        }
        std::vector<uint64_t> edges(uniq.begin(), uniq.end());
        long m = static_cast<long>(edges.size());
        if (m == 0) continue;
        std::shuffle(edges.begin(), edges.end(), rng);
        ++checked;

        int nbatch = 1 + rng() % 5;
        long bs = (m + nbatch - 1) / nbatch;

        // ---- ground truth -----------------------------------------------
        long ref_bridges = 0, ref_iso = 0;
        long ref_count = serial_bcc_count(edges.data(), m, n, &ref_bridges, &ref_iso);

        std::vector<int>  ref_beta;
        std::vector<char> ref_cut;
        std::vector<long> ref_sizes;
        serial_bcc_partition(edges.data(), m, n, ref_beta, ref_cut, ref_sizes);

        // vertices of each true BCC
        std::map<int, std::set<int>> bcc_verts;
        for (long e = 0; e < m; ++e) {
            bcc_verts[ref_beta[e]].insert(static_cast<int>(edges[e] >> 32));
            bcc_verts[ref_beta[e]].insert(static_cast<int>(edges[e] & 0xFFFFFFFF));
        }

        if (getenv("BCC_EDGES")) {
            printf("G n=%d m=%ld nbatch=%d bs=%ld\nedges:", n, m, nbatch, bs);
            for (uint64_t e : edges)
                printf(" (%d,%d)", (int)(e >> 32), (int)(e & 0xFFFFFFFF));
            printf("\ntruth bccs:\n");
            for (auto& kv : bcc_verts) {
                printf("  %d:", kv.first);
                for (int v : kv.second) printf(" %d", v);
                printf("\n");
            }
        }

        // ---- stream it through ------------------------------------------
        GPU_BCG g(n, m, bs);
        g.numEdges = 0;
        for (long off = 0; off < m; off += bs) {
            long cur = std::min(bs, m - off);
            CUDA_CHECK(cudaMemcpy(g.d_edgelist, edges.data() + off, cur * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice), "upload batch");
            g_bcc_last_batch = (off + cur >= m);
            single_pass_batch_gpu(g, cur);
            {
                // A block must own SOMETHING: at least one resident, or at
                // least one incident edge in H. A block with neither names an
                // empty vertex set and would be counted as a BCC that does not
                // exist. Reachable only from uninitialised state (that was the
                // stale d_blk_flag[n_t] sentinel bug), never from de-merging --
                // de-merge always leaves the leaver's link edge behind. This
                // check is here to keep that true.
                std::vector<int> Mb(n);
                CUDA_CHECK(cudaMemcpy(Mb.data(), g.d_mapping, n*sizeof(int),
                                      cudaMemcpyDeviceToHost), "Mb");
                std::vector<uint64_t> eb(g.numEdges);
                if (g.numEdges)
                    CUDA_CHECK(cudaMemcpy(eb.data(), g.updated_edgelist,
                               g.numEdges*sizeof(uint64_t), cudaMemcpyDeviceToHost), "eb");
                std::vector<int> res(g.n_b, 0), inc(g.n_b, 0);
                for (int v = 0; v < n; ++v)
                    if (Mb[v] >= 0 && Mb[v] < g.n_b) ++res[Mb[v]];
                for (uint64_t e : eb) {
                    int x = (int)(e >> 32), y = (int)(e & 0xFFFFFFFF);
                    if (x < g.n_b) ++inc[x];
                    if (y < g.n_b) ++inc[y];
                }
                // THIN BLOCK: 0 residents and exactly 2 incident edges in H.
                // A legitimate block is a BCC, so it has either >= 1 non-cut
                // member (a resident) or >= 3 cut members (>= 3 incident
                // edges) -- a BCC with exactly two members and no residents is
                // a single edge, i.e. a bridge, and bridges are never blocks.
                // So a thin block is ALWAYS spurious, which makes it a
                // candidate signature for an end-of-stream repair.
                for (int bb = 0; bb < g.n_b; ++bb)
                    if (res[bb] == 0 && inc[bb] == 2) { ++fail_thin; break; }

                for (int bb = 0; bb < g.n_b; ++bb)
                    if (res[bb] == 0 && inc[bb] == 0) {
                        printf("trial %d: EMPTY BLOCK %d (no residents, no edges) "
                               "n=%d m=%ld batches=%d off=%ld n_s=%d n_b=%d\n",
                               t, bb, n, m, nbatch, off, g.n_s, g.n_b);
                        ++fail_empty;
                        break;
                    }

                // NO FALSE CUT VERTEX. Every OWN-node of H must hold exactly
                // one original vertex, i.e. |f^-1(x)| == 1 for x >= n_b. This
                // is what makes Euro-Par's end-of-stream BCC_Repair -- whose
                // whole job is the d_counter[v] == 1 test on false cut
                // vertices -- unnecessary here: with the per-round filter
                // there are never any false cut vertices to repair.
                {
                    std::vector<int> own(g.n_s - g.n_b, 0);
                    for (int v = 0; v < n; ++v)
                        if (Mb[v] >= g.n_b && Mb[v] < g.n_s) ++own[Mb[v] - g.n_b];
                    for (int x = 0; x < (int)own.size(); ++x)
                        if (own[x] > 1) {
                            printf("trial %d: FALSE CUT own-node %d holds %d verts "
                                   "n=%d m=%ld batches=%d off=%ld n_s=%d n_b=%d\n",
                                   t, x + g.n_b, own[x], n, m, nbatch, off,
                                   g.n_s, g.n_b);
                            ++fail_false;
                            break;
                        }
                }
            }
        }

        std::vector<int> M(n);
        CUDA_CHECK(cudaMemcpy(M.data(), g.d_mapping, n * sizeof(int), cudaMemcpyDeviceToHost),
                   "download M");
        std::vector<uint64_t> fin(g.numEdges);
        if (g.numEdges)
            CUDA_CHECK(cudaMemcpy(fin.data(), g.updated_edgelist, g.numEdges * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost), "download final edges");

        // ---- 1. count ---------------------------------------------------
        const int two_nb = g.n_b;
        long bridges = 0;
        for (uint64_t e : fin) {
            int x = static_cast<int>(e >> 32), y = static_cast<int>(e & 0xFFFFFFFF);
            if (x >= two_nb && y >= two_nb) ++bridges;
        }
        long got_count = g.n_b + bridges;
        if (got_count != ref_count) {
            printf("trial %d: COUNT n=%d m=%ld batches=%d  got %ld (%d blocks + %ld bridges)"
                   "  want %ld (%ld bridges)\n",
                   t, n, m, nbatch, got_count, g.n_b, bridges, ref_count, ref_bridges);
            ++fail_count;
            if (getenv("BCC_DUMP")) {
                printf("  n_s=%d n_b=%d two_nb=%d  final |E|=%ld\n",
                       g.n_s, g.n_b, two_nb, (long)g.numEdges);
                printf("  final edges:");
                for (uint64_t e : fin)
                    printf(" (%d,%d)", (int)(e >> 32), (int)(e & 0xFFFFFFFF));
                printf("\n  M[v] for v with M[v] < two_nb (inside a block):\n   ");
                for (int v = 0; v < n; ++v) if (M[v] < two_nb) printf(" %d->%d", v, M[v]);
                printf("\n  truth: %zu BCCs\n", bcc_verts.size());
                for (auto& kv : bcc_verts) {
                    printf("    bcc %d (%zu verts):", kv.first, kv.second.size());
                    for (int v : kv.second) printf(" %d", v);
                    printf("\n");
                }
            }
        }

        // ---- 2. no real BCC split ---------------------------------------
        for (auto& [j, vs] : bcc_verts) {
            if (vs.size() <= 2) continue;              // a bridge is not compressed
            std::vector<int> nc;
            for (int v : vs) if (!ref_cut[v]) nc.push_back(v);
            for (size_t i = 1; i < nc.size(); ++i) {
                if (M[nc[0]] != M[nc[i]]) {
                    printf("trial %d: SPLIT n=%d m=%ld batches=%d  bcc %d: %d->%d but %d->%d\n",
                           t, n, m, nbatch, j, nc[0], M[nc[0]], nc[i], M[nc[i]]);
                    ++fail_split;
                    i = nc.size();
                }
            }
        }

        // ---- 3. no over-merge -------------------------------------------
        std::map<int, std::set<int>> share;   // M value -> vertices
        for (int v = 0; v < n; ++v) share[M[v]].insert(v);
        for (auto& [s, vs] : share) {
            if (vs.size() <= 1) continue;
            // everyone sharing a supernode must sit in one common true BCC
            bool ok = false;
            for (auto& [j, bv] : bcc_verts) {
                bool all = true;
                for (int v : vs) if (!bv.count(v)) { all = false; break; }
                if (all) { ok = true; break; }
            }
            if (!ok) {
                printf("trial %d: OVER-MERGE n=%d m=%ld batches=%d  supernode %d holds %zu verts"
                       " with no common BCC\n", t, n, m, nbatch, s, vs.size());
                ++fail_merge;
                break;
            }
        }

        if (fail_count + fail_split + fail_merge > 6) break;
    }

    printf("\n  graphs checked               : %d\n", checked);
    printf("  BCC count vs serial          : %s\n", fail_count ? "FAIL" : "all exact");
    printf("  real BCC split               : %d\n", fail_split);
    printf("  over-merge                   : %d\n", fail_merge);
    printf("  empty blocks (no res,no edge): %s\n", fail_empty ? "FOUND" : "none");
    printf("  false cut (|f^-1(own)| != 1)  : %s\n", fail_false ? "FOUND" : "none");
    printf("  thin blocks (0 res, 2 edges)  : %s\n", fail_thin ? "FOUND" : "none");
    return (fail_count || fail_split || fail_merge) ? 1 : 0;
}
