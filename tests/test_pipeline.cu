//---------------------------------------------------------------------
// End-to-end test: the single-pass GPU pipeline (phases 3-4-5) against the
// serial prototype, over the same graph and the same batching.
//
// Compares the PARTITION of original vertices, not raw ids -- the two use
// different id schemes by construction.
//
// Build:
//   nvcc -std=c++17 -O2 -I include -I ../prototype \
//        tests/test_pipeline.cu src/extern_bcc/single_pass.cu \
//        src/extern_bcc/serial_bcc.cu src/extern_bcc/bcg_memory_utils.cu \
//        src/extern_bcc/edge_cleanup.cu -o build/test_pipeline
//---------------------------------------------------------------------

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>

#include <cuda_runtime.h>

#include "core_fix.hpp"
#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"
#include "extern_bcc/single_pass.cuh"

bool g_verbose          = false;
int  local_block_size   = 1000;
long maxThreadsPerBlock = 0;

static uint64_t pack(int a, int b) {
    int lo = std::min(a, b), hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint32_t>(hi);
}

// Canonical form of a partition: for each element, the index of the first
// element sharing its group. Independent of the label values used.
static std::vector<int> canon(const std::vector<int>& lbl) {
    std::map<int,int> first;
    std::vector<int> c(lbl.size());
    for (size_t i = 0; i < lbl.size(); ++i) {
        auto it = first.find(lbl[i]);
        if (it == first.end()) { first[lbl[i]] = (int)i; c[i] = (int)i; }
        else c[i] = it->second;
    }
    return c;
}

int main(int argc, char** argv) {
    int seed0  = (argc > 1) ? atoi(argv[1]) : 1;
    int trials = (argc > 2) ? atoi(argv[2]) : 300;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0), "failed to query device");
    maxThreadsPerBlock = prop.maxThreadsPerBlock;

    const int  N  = 64;
    const long BS = 64;

    GPU_BCG g(N, 8 * BS, BS);

    struct Res { int ap_lost = 0, over_merge = 0, split_real = 0, split_bridge = 0; bool report = false; int shown = 0; };
    Res rc, rg;   // prototype, GPU
    rg.report = true;

    int checked = 0, mismatch = 0;

    for (int t = 0; t < trials; ++t) {
        std::mt19937 rng(seed0 + t);

        const int n = 6 + rng() % 10;
        std::vector<std::pair<int64_t,int64_t>> all;
        for (int u = 0; u < n; ++u)
            for (int v = u + 1; v < n; ++v)
                if ((int)(rng() % 100) < 35) all.push_back({u, v});
        if (all.size() < 3) continue;
        checked++;

        std::shuffle(all.begin(), all.end(), rng);
        const int nb = 1 + rng() % 3;
        std::vector<std::vector<std::pair<int64_t,int64_t>>> batches(nb);
        for (size_t i = 0; i < all.size(); ++i) batches[i % nb].push_back(all[i]);

        // ---- CPU prototype ----
        single_tour_external_memory_bcc S(n, all.size());
        for (auto& b : batches) S.processBatch(b);
        std::vector<int> cpu(n);
        for (int i = 0; i < n; ++i) cpu[i] = (int)S.resolve(i);

        // ---- GPU pipeline ----
        // Identity mapping, and an EMPTY active edge set: there is no pass-1
        // spanning tree any more, so numEdges starts at 0 rather than n-1.
        std::vector<int> ident(N);
        for (int i = 0; i < N; ++i) ident[i] = i;
        CUDA_CHECK(cudaMemcpy(g.d_mapping, ident.data(), N * sizeof(int), cudaMemcpyHostToDevice), "init mapping");
        g.numEdges = 0;

        for (auto& b : batches) {
            std::vector<uint64_t> packed;
            for (auto& [u, v] : b) packed.push_back(pack((int)u, (int)v));
            CUDA_CHECK(cudaMemcpy(g.d_edgelist, packed.data(), packed.size() * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice), "upload batch");
            single_pass_batch(g, (long)packed.size());
        }

        std::vector<int> gm(N);
        CUDA_CHECK(cudaMemcpy(gm.data(), g.d_mapping, N * sizeof(int), cudaMemcpyDeviceToHost), "read mapping");
        std::vector<int> gpu(gm.begin(), gm.begin() + n);

        if (canon(cpu) != canon(gpu)) mismatch++;

        // ---- ground truth: Tarjan over the whole graph ----
        std::vector<std::vector<int64_t>> adj(n);
        for (auto& [u, v] : all) { adj[u].push_back(v); adj[v].push_back(u); }
        std::vector<bool> vis(n, false);
        std::vector<int64_t> hi(n, 0), dep(n, 0), apv;
        std::vector<std::vector<std::pair<int64_t,int64_t>>> bccs;
        std::stack<std::pair<int64_t,int64_t>, std::vector<std::pair<int64_t,int64_t>>> stk;
        for (int i = 0; i < n; ++i)
            if (!vis[i]) algorithm::bcc(adj, vis, hi, dep, apv, bccs, stk, 0, i, -1);
        std::set<int64_t> ap(apv.begin(), apv.end());
        std::vector<std::set<int64_t>> comps;
        for (auto& c : bccs) {
            std::set<int64_t> s;
            for (auto& [u, v] : c) { s.insert(u); s.insert(v); }
            comps.push_back(s);
        }

        // fuzz5's TEST A / B / C, applied to whichever mapping is passed in
        auto score = [&](const std::vector<int>& M, Res& r) {
            for (int64_t a : ap) if (M[a] != (int)a) { r.ap_lost++; break; }

            bool bad = false;
            int bu = -1, bv = -1;
            for (int u = 0; u < n && !bad; ++u)
                for (int v = u + 1; v < n && !bad; ++v)
                    if (M[u] == M[v]) {
                        if (ap.count(u) || ap.count(v)) { bad = true; bu = u; bv = v; break; }
                        bool together = false;
                        for (auto& c : comps) if (c.count(u) && c.count(v)) together = true;
                        if (!together) { bad = true; bu = u; bv = v; }
                    }
            if (bad) {
                r.over_merge++;
                if (r.report && r.shown < 3) {
                    r.shown++;
                    printf("OVER-MERGE seed=%d n=%d nb=%d : %d and %d both -> %d\n",
                           seed0 + t, n, nb, bu, bv, M[bu]);
                    printf("  edges:"); for (auto& [x,y] : all) printf(" (%ld,%ld)", (long)x, (long)y);
                    printf("\n  truth AP:"); for (int64_t a : ap) printf(" %ld", (long)a);
                    printf("\n  truth BCCs:");
                    for (auto& c : comps) { printf(" {"); for (int64_t x : c) printf("%ld ", (long)x); printf("}"); }
                    printf("\n  gpu map :"); for (int i = 0; i < n; ++i) printf(" %d->%d", i, M[i]);
                    printf("\n");
                }
            }

            bool missC = false, missBridge = false;
            for (auto& c : comps) {
                std::vector<int64_t> nc;
                for (int64_t x : c) if (!ap.count(x)) nc.push_back(x);
                for (size_t i = 1; i < nc.size(); ++i)
                    if (M[nc[0]] != M[nc[i]]) { if (c.size() <= 2) missBridge = true; else missC = true; }
            }
            if (missC) r.split_real++;
            if (missBridge) r.split_bridge++;
        };

        score(cpu, rc);
        score(gpu, rg);
    }

    printf("\n  graphs checked             : %d\n", checked);
    printf("  CPU/GPU partition differs  : %d\n\n", mismatch);
    printf("  vs Tarjan ground truth      %10s %10s\n", "prototype", "GPU");
    printf("  articulation points lost   %10d %10d\n", rc.ap_lost,      rg.ap_lost);
    printf("  over-merges                %10d %10d\n", rc.over_merge,   rg.over_merge);
    printf("  BCC split, real BCC        %10d %10d\n", rc.split_real,   rg.split_real);
    printf("  BCC split, bridge only     %10d %10d\n", rc.split_bridge, rg.split_bridge);
    return 0;
}
