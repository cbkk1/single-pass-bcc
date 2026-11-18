//---------------------------------------------------------------------
// Verifies the iterative phase-4 BCC against the recursive reference in the
// serial prototype (algorithm::bcc in core_fix.hpp).
//
// Checks, on random graphs including forests and isolated nodes:
//   1. the articulation-point set matches exactly
//   2. the partition induced by I[] matches the reference BCC partition
//   3. bridges are left uncompressed (I[s] == s), as designed
//
// Build:
//   nvcc -std=c++17 -O2 -I include -I ../prototype \
//        tests/test_serial_bcc.cu src/extern_bcc/serial_bcc.cu \
//        -o build/test_serial_bcc
//---------------------------------------------------------------------

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "core_fix.hpp"                    // the recursive reference
#include "extern_bcc/serial_bcc.cuh"

static uint64_t pack(int a, int b) {
    int lo = std::min(a, b), hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint32_t>(hi);
}

int main(int argc, char** argv) {
    int seed0  = (argc > 1) ? atoi(argv[1]) : 1;
    int trials = (argc > 2) ? atoi(argv[2]) : 500;

    int fail_ap = 0, fail_part = 0, fail_bridge = 0, fail_count = 0, checked = 0;
    int shown = 0;

    for (int t = 0; t < trials; ++t) {
        std::mt19937 rng(seed0 + t);

        const int N = 5 + rng() % 25;
        // Low density gives forests, bridges and isolated vertices -- exactly
        // the cases the two-pass code never had to handle.
        const int density = 8 + rng() % 22;

        std::set<uint64_t> eset;
        for (int u = 0; u < N; ++u)
            for (int v = u + 1; v < N; ++v)
                if ((int)(rng() % 100) < density) eset.insert(pack(u, v));
        if (eset.empty()) continue;
        checked++;

        std::vector<uint64_t> edges(eset.begin(), eset.end());

        // ---- reference: recursive Tarjan from the prototype ----
        std::vector<std::vector<int64_t>> adj(N);
        for (uint64_t e : edges) {
            int u = e >> 32, v = e & 0xFFFFFFFF;
            adj[u].push_back(v); adj[v].push_back(u);
        }
        std::vector<bool> vis(N, false);
        std::vector<int64_t> hi(N, 0), dep(N, 0), apv;
        std::vector<std::vector<std::pair<int64_t,int64_t>>> ref_bccs;
        std::stack<std::pair<int64_t,int64_t>, std::vector<std::pair<int64_t,int64_t>>> stk;
        for (int i = 0; i < N; ++i)
            if (!vis[i]) algorithm::bcc(adj, vis, hi, dep, apv, ref_bccs, stk, 0, i, -1);

        std::set<int> ref_ap(apv.begin(), apv.end());

        // ---- 0. exact BCC count (the -c checker path) ----
        {
            long got_bridges = 0, got_iso = 0;
            long got_count = serial_bcc_count((const uint64_t*)edges.data(),
                                              (long)edges.size(), N,
                                              &got_bridges, &got_iso);

            long want_bridges = 0;
            for (auto& c : ref_bccs) if (c.size() < 2) want_bridges++;

            long want_iso = 0;
            for (int i = 0; i < N; ++i) if (adj[i].empty()) want_iso++;

            if (got_count != (long)ref_bccs.size() || got_bridges != want_bridges
                || got_iso != want_iso) {
                fail_count++;
                if (shown < 3) {
                    shown++;
                    printf("trial %d (N=%d): COUNT MISMATCH\n"
                           "  want: bcc=%zu bridges=%ld isolated=%ld\n"
                           "  got : bcc=%ld bridges=%ld isolated=%ld\n",
                           t, N, ref_bccs.size(), want_bridges, want_iso,
                           got_count, got_bridges, got_iso);
                }
            }
        }

        // ---- subject: iterative, over the same id space ----
        std::vector<int> I;
        std::vector<char> iscut;
        // Block ids are minted above N, so the id space must extend past it.
        std::vector<char> live(2 * N, 0);
        serial_bcc(edges, N, 2 * N, live, I, iscut);

        std::set<int> got_ap;
        for (int i = 0; i < N; ++i) if (iscut[i]) got_ap.insert(i);

        // ---- 1. articulation points ----
        if (got_ap != ref_ap) {
            fail_ap++;
            if (shown < 3) {
                shown++;
                printf("trial %d (N=%d): AP MISMATCH\n  ref:", t, N);
                for (int a : ref_ap) printf(" %d", a);
                printf("\n  got:");
                for (int a : got_ap) printf(" %d", a);
                printf("\n");
            }
            continue;
        }

        // ---- expected partition from the reference ----
        // Non-cut vertices of a component with >= 2 edges share a label;
        // everything else stands alone.
        std::vector<int> want(N);
        for (int i = 0; i < N; ++i) want[i] = i;
        int lbl = N;
        bool bridge_ok = true;
        for (auto& c : ref_bccs) {
            std::set<int64_t> verts;
            for (auto& [a, b] : c) { verts.insert(a); verts.insert(b); }
            if (c.size() < 2) {
                // bridge: must be left uncompressed
                for (int64_t x : verts)
                    if (!ref_ap.count((int)x) && I[x] != (int)x) bridge_ok = false;
                continue;
            }
            bool any = false;
            for (int64_t x : verts) if (!ref_ap.count((int)x)) any = true;
            if (!any) continue;
            for (int64_t x : verts) if (!ref_ap.count((int)x)) want[x] = lbl;
            lbl++;
        }
        if (!bridge_ok) fail_bridge++;

        // ---- 2. partition equality (label values differ; grouping must not) ----
        std::map<int,int> f, g;
        bool same = true;
        for (int i = 0; i < N && same; ++i) {
            auto itf = f.find(want[i]);
            auto itg = g.find(I[i]);
            if ((itf == f.end()) != (itg == g.end())) { same = false; break; }
            if (itf == f.end()) { f[want[i]] = I[i]; g[I[i]] = want[i]; }
            else if (itf->second != I[i] || itg->second != want[i]) same = false;
        }
        if (!same) {
            fail_part++;
            if (shown < 3) {
                shown++;
                printf("trial %d (N=%d): PARTITION MISMATCH\n  want:", t, N);
                for (int i = 0; i < N; ++i) printf(" %d->%d", i, want[i]);
                printf("\n  got :");
                for (int i = 0; i < N; ++i) printf(" %d->%d", i, I[i]);
                printf("\n");
            }
        }
    }

    printf("\n  graphs checked            : %d\n", checked);
    printf("  exact BCC count (-c path) : %s\n", fail_count ? "FAIL" : "matches reference");
    printf("  articulation-point set    : %s\n", fail_ap    ? "FAIL" : "matches reference");
    printf("  BCC partition             : %s\n", fail_part  ? "FAIL" : "matches reference");
    printf("  bridges left uncompressed : %s\n", fail_bridge? "FAIL" : "as designed");
    return (fail_ap || fail_part || fail_bridge || fail_count) ? 1 : 0;
}
