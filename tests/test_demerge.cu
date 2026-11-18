//---------------------------------------------------------------------
// Differential test for the single-pass de-merger (phase 3).
//
// Verifies the CUDA de-merger against a CPU reference on random inputs,
// WITHOUT needing phase 4. Compares the emitted edge set and the resulting
// supernode mapping.
//
// Build:
//   nvcc -std=c++17 -O2 -I include tests/test_demerge.cu \
//        src/extern_bcc/single_pass.cu src/extern_bcc/serial_bcc.cu \
//        src/extern_bcc/bcg_memory_utils.cu src/extern_bcc/edge_cleanup.cu \
//        -o build/test_demerge
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

// Globals normally defined in main.cu
bool g_verbose          = false;
int  local_block_size   = 1000;
long maxThreadsPerBlock = 0;

static uint64_t pack(int a, int b) {
    int lo = std::min(a, b), hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint32_t>(hi);
}

// PARALLEL semantics -- what the paper specifies and what the GPU does.
// Every edge reads the pre-batch mapping; extractions apply afterwards.
//
// A leaver joins BOTH halves of the block it came from. That is what keeps it
// 2-connected to the remnant, and any two leavers of one block 2-connected to
// each other; a single edge to the block would be a bridge and lose both.
static std::set<uint64_t> cpu_demerge_parallel(
    int n, std::vector<int>& M, const std::vector<std::pair<int,int>>& edges) {

    std::set<uint64_t> out;
    std::vector<int> extract(n, 0);

    for (auto [u, v] : edges) {
        int mu = M[u], mv = M[v];
        if (mu == mv) continue;
        if (mu >= n) { out.insert(pack(u, mu)); out.insert(pack(u, mu + 1)); extract[u] = 1; }
        if (mv >= n) { out.insert(pack(v, mv)); out.insert(pack(v, mv + 1)); extract[v] = 1; }
        out.insert(pack(u, v));
    }
    for (int i = 0; i < n; ++i)
        if (extract[i]) M[i] = i;

    return out;
}

int main(int argc, char** argv) {
    int seed0  = (argc > 1) ? atoi(argv[1]) : 1;
    int trials = (argc > 2) ? atoi(argv[2]) : 200;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0), "failed to query device");
    maxThreadsPerBlock = prop.maxThreadsPerBlock;

    const int  N  = 512;    // fixed vertex count; demerge() reads orig_numVert
    const long BS = 256;    // max batch size

    GPU_BCG g(N, 4 * BS, BS);

    uint64_t* d_out = g.d_edge_buffer;

    int failures = 0;

    for (int t = 0; t < trials; ++t) {
        std::mt19937 rng(seed0 + t);

        // ---- build a mapping in split-range form -------------------------
        // [0, N) vertices by index, [N, 2N) blocks.
        int nblocks = 1 + rng() % (N / 4);
        std::vector<int> M(N);
        for (int i = 0; i < N; ++i)
            // Blocks own ALIGNED PAIRS above N, and a resident always maps to
            // the primary half, so the id is N plus an even offset.
            M[i] = (rng() % 100 < 45) ? (N + 2 * (int)(rng() % nblocks)) : i;

        // ---- random batch ------------------------------------------------
        long bs = 1 + rng() % BS;
        std::vector<std::pair<int,int>> edges;
        std::vector<uint64_t> packed;
        for (long i = 0; i < bs; ++i) {
            int u = rng() % N, v = rng() % N;
            if (u == v) { v = (v + 1) % N; }
            edges.push_back({u, v});
            packed.push_back(pack(u, v));
        }

        // ---- CPU references ---------------------------------------------
        std::vector<int> M_par = M;
        std::set<uint64_t> want = cpu_demerge_parallel(N, M_par, edges);

        // ---- GPU ---------------------------------------------------------
        CUDA_CHECK(cudaMemcpy(g.d_mapping, M.data(), N * sizeof(int), cudaMemcpyHostToDevice), "upload M");
        CUDA_CHECK(cudaMemcpy(g.d_edgelist, packed.data(), bs * sizeof(uint64_t), cudaMemcpyHostToDevice), "upload edges");

        long total = demerge(g, bs, d_out, 0);

        std::vector<uint64_t> got_raw(total);
        if (total) CUDA_CHECK(cudaMemcpy(got_raw.data(), d_out, total * sizeof(uint64_t), cudaMemcpyDeviceToHost), "download edges");
        std::set<uint64_t> got(got_raw.begin(), got_raw.end());

        std::vector<int> M_gpu(N);
        CUDA_CHECK(cudaMemcpy(M_gpu.data(), g.d_mapping, N * sizeof(int), cudaMemcpyDeviceToHost), "download M");

        // ---- compare -----------------------------------------------------
        bool bad = false;
        if (got != want) {
            printf("trial %d: EDGE SET MISMATCH (gpu %zu vs cpu %zu)\n", t, got.size(), want.size());
            bad = true;
        }
        if (M_gpu != M_par) {
            printf("trial %d: MAPPING MISMATCH\n", t);
            bad = true;
        }
        if (bad && ++failures > 3) break;
    }

    printf("\n  trials                       : %d\n", trials);
    printf("  GPU vs CPU (parallel sem.)   : %s\n", failures ? "FAIL" : "all match");
    return failures ? 1 : 0;
}
