//---------------------------------------------------------------------
// Differential test for GPU Compress-Graph (Algorithm 4).
//
// Checks the two things that are genuinely new, against the serial Tarjan:
//
//   1. the per-edge block PARTITION -- i.e. that beta groups edges exactly as
//      Tarjan's biconnected components do. Ids differ, so this compares the
//      partition, via a bijection built edge by edge.
//   2. the cut-vertex set, as a set of node ids.
//
// Both are checked on DISCONNECTED inputs, because that is what the single-pass
// active edge set looks like and what Fast_BCC cannot take unaugmented.
//
// Build:
//   nvcc -std=c++17 -O2 -I include tests/test_compress.cu \
//        src/extern_bcc/gpu_compress.cu src/extern_bcc/single_pass.cu \
//        src/extern_bcc/serial_bcc.cu src/extern_bcc/bcg_memory_utils.cu \
//        src/extern_bcc/edge_cleanup.cu src/fast_bcc/*.cu \
//        src/utility/cuda_utility.cu -o build/test_compress
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
#include "extern_bcc/serial_bcc.cuh"

bool g_verbose          = false;
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

    // N must exceed LOCAL_BLOCK_SIZE (1000) in the Euler position space, i.e.
    // 2N > 1000, or the sparse-table RMQ degenerates to a single block and is
    // never actually exercised. At N = 256 it wasn't, which hid three launch-
    // bound bugs that only appear on a forest.
    const int  N  = 4000;
    const long BS = 512;
    GPU_BCG g(N, 8 * BS, BS);

    int fail_part = 0, fail_cut = 0, checked = 0;

    for (int t = 0; t < trials; ++t) {
        std::mt19937 rng(seed0 + t);

        // A sparse random graph on a random prefix of the id space, so most
        // trials are a forest of several components with isolated nodes left
        // over -- exactly the shape Fast_BCC cannot take unaugmented.
        int  n_t = 4 + rng() % (N - 4);
        long m_a = 1 + rng() % (2 * n_t);

        std::set<uint64_t> uniq;
        for (long i = 0; i < m_a; ++i) {
            int u = rng() % n_t, v = rng() % n_t;
            if (u != v) uniq.insert(pk(u, v));
        }
        std::vector<uint64_t> edges(uniq.begin(), uniq.end());
        m_a = static_cast<long>(edges.size());
        if (m_a == 0) continue;
        ++checked;

        // ---- serial reference -------------------------------------------
        std::vector<int>  ref_beta;
        std::vector<char> ref_cut;
        std::vector<long> ref_sizes;
        serial_bcc_partition(edges.data(), m_a, n_t, ref_beta, ref_cut, ref_sizes);

        // ---- GPU ---------------------------------------------------------
        CUDA_CHECK(cudaMemcpy(g.updated_edgelist, edges.data(), m_a * sizeof(uint64_t),
                              cudaMemcpyHostToDevice), "upload edges");
        bcc_labels_gpu(g, n_t, m_a);

        std::vector<int> got_beta(m_a), got_cut(n_t);
        CUDA_CHECK(cudaMemcpy(got_beta.data(), g.d_beta, m_a * sizeof(int),
                              cudaMemcpyDeviceToHost), "download beta");
        CUDA_CHECK(cudaMemcpy(got_cut.data(), g.iscutVertex, n_t * sizeof(int),
                              cudaMemcpyDeviceToHost), "download cut");

        // ---- compare the partition, not the ids --------------------------
        std::map<int, int> fwd, rev;
        bool bad = false;
        for (long e = 0; e < m_a && !bad; ++e) {
            auto f = fwd.emplace(got_beta[e], ref_beta[e]);
            if (!f.second && f.first->second != ref_beta[e]) bad = true;  // GPU merged too much
            auto r = rev.emplace(ref_beta[e], got_beta[e]);
            if (!r.second && r.first->second != got_beta[e]) bad = true;  // GPU split too much
        }
        if (bad) {
            printf("trial %d: BLOCK PARTITION MISMATCH (n_t=%d m_a=%d)\n", t, n_t, (int)m_a);
            ++fail_part;
        }

        for (int v = 0; v < n_t; ++v) {
            if ((got_cut[v] != 0) != (ref_cut[v] != 0)) {
                printf("trial %d: CUT MISMATCH at %d (gpu %d, ref %d)  n_t=%d m_a=%d\n",
                       t, v, got_cut[v], (int)ref_cut[v], n_t, (int)m_a);
                ++fail_cut;
                break;
            }
        }

        if (fail_part + fail_cut > 4) break;
    }

    printf("\n  graphs checked               : %d\n", checked);
    printf("  edge -> block partition      : %s\n", fail_part ? "FAIL" : "all match");
    printf("  cut-vertex set               : %s\n", fail_cut  ? "FAIL" : "all match");
    return (fail_part || fail_cut) ? 1 : 0;
}
