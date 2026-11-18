//---------------------------------------------------------------------
// Differential test for the LCH readout.
//
// Checks the defining identity against the serial Tarjan:
//
//     BCC(j)  ==  { v : label[v] = j }  u  { componentHead[j] }
//
// Compared as a SET OF VERTEX SETS, so component ids never have to agree --
// only the decomposition does.
//
// Build:
//   nvcc -std=c++17 -O2 --expt-relaxed-constexpr -I include tests/test_lch.cu \
//        src/extern_bcc/lch.cu src/extern_bcc/gpu_compress.cu \
//        src/extern_bcc/single_pass.cu src/extern_bcc/serial_bcc.cu \
//        src/extern_bcc/bcg_memory_utils.cu src/extern_bcc/edge_cleanup.cu \
//        src/fast_bcc/*.cu src/utility/cuda_utility.cu -o build/test_lch
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
#include "extern_bcc/single_pass.cuh"
#include "extern_bcc/serial_bcc.cuh"
#include "extern_bcc/lch.cuh"

extern bool g_bcc_last_batch;
bool g_verbose          = false;
bool g_gpu_bcc          = true;
int  local_block_size   = 1000;
long maxThreadsPerBlock = 0;

static uint64_t pk(int a, int b) {
    int lo = std::min(a, b), hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint32_t>(hi);
}

int main(int argc, char** argv) {
    int seed0  = (argc > 1) ? atoi(argv[1]) : 1;
    int trials = (argc > 2) ? atoi(argv[2]) : 300;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0), "failed to query device");
    maxThreadsPerBlock = prop.maxThreadsPerBlock;

    const int N = 64;
    int failures = 0, checked = 0;
    int bad_head = 0, names_nothing = 0, blk_blk = 0;

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

        int nbatch = 1 + rng() % 4;
        long bs = (m + nbatch - 1) / nbatch;

        // ---- truth: vertex set of every BCC ------------------------------
        std::vector<int>  ref_beta;
        std::vector<char> ref_cut;
        std::vector<long> ref_sz;
        serial_bcc_partition(edges.data(), m, n, ref_beta, ref_cut, ref_sz);

        std::map<int, std::set<int>> truth_map;
        for (long e = 0; e < m; ++e) {
            truth_map[ref_beta[e]].insert((int)(edges[e] >> 32));
            truth_map[ref_beta[e]].insert((int)(edges[e] & 0xFFFFFFFF));
        }
        std::set<std::set<int>> truth;
        for (auto& [j, s] : truth_map) truth.insert(s);

        // ---- stream it through -------------------------------------------
        GPU_BCG g(n, m, bs);
        g.numEdges = 0;
        for (long off = 0; off < m; off += bs) {
            long cur = std::min(bs, m - off);
            CUDA_CHECK(cudaMemcpy(g.d_edgelist, edges.data() + off, cur * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice), "upload batch");
            g_bcc_last_batch = (off + cur >= m);
            single_pass_batch_gpu(g, cur);
        }

        std::vector<int> M(n);
        CUDA_CHECK(cudaMemcpy(M.data(), g.d_mapping, n * sizeof(int), cudaMemcpyDeviceToHost), "M");
        std::vector<uint64_t> fin(g.numEdges);
        if (g.numEdges)
            CUDA_CHECK(cudaMemcpy(fin.data(), g.updated_edgelist, g.numEdges * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost), "edges");

        // ---- LCH ----------------------------------------------------------
        std::vector<int> label, head;
        int n_comp = compute_lch(M, n, fin, g.n_s, g.n_b, label, head);

        // ---- rebuild the decomposition from (label, head) -----------------
        std::vector<std::set<int>> got_v(n_comp);
        for (int v = 0; v < n; ++v)
            if (label[v] >= 0) got_v[label[v]].insert(v);
        for (int j = 0; j < n_comp; ++j)
            if (head[j] >= 0) got_v[j].insert(head[j]);

        // ---- the two concerns about EMPTY blocks ---------------------------
        // (a) a head must be a REAL original vertex. An empty block node has no
        //     resident, so if one ever became a head we would report a head that
        //     corresponds to no vertex of G.
        // (b) a label whose class is empty AND which has no head names nothing,
        //     i.e. it would be counted as a bcc that does not exist.
        // (c) H must contain no block-block edge: a block's parent in the
        //     block-cut forest has to be a cut vertex, never another block.
        for (int j = 0; j < n_comp; ++j) {
            if (head[j] != -1 && (head[j] < 0 || head[j] >= n)) { ++bad_head; break; }
            if (got_v[j].empty() && head[j] < 0)                { ++names_nothing; break; }
        }
        for (uint64_t e : fin) {
            int x = (int)(e >> 32), y = (int)(e & 0xFFFFFFFF);
            if (x < g.n_b && y < g.n_b) { ++blk_blk; break; }
        }

        std::set<std::set<int>> got;
        for (auto& s : got_v) if (s.size() >= 2) got.insert(s);   // a lone vertex is no BCC

        if (got != truth) {
            printf("trial %d: LCH MISMATCH  n=%d m=%ld batches=%d  (got %zu comps, want %zu)\n",
                   t, n, m, nbatch, got.size(), truth.size());
            if (++failures > 3) break;
        }
    }

    printf("\n  graphs checked               : %d\n", checked);
    printf("  BCC(j) = class(j) u head(j)  : %s\n", failures ? "FAIL" : "all match");
    printf("  head is a real vertex        : %s\n", bad_head      ? "VIOLATED" : "always");
    printf("  label naming nothing         : %s\n", names_nothing ? "FOUND"    : "none");
    printf("  block-block edge in H        : %s\n", blk_blk       ? "FOUND"    : "none");
    return failures ? 1 : 0;
}
