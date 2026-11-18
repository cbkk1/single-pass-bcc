// Run one hand-written graph through the pipeline and diff against serial Tarjan.
//   ./case <nbatch> <n> u1 v1 u2 v2 ...
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"
#include "extern_bcc/single_pass.cuh"
#include "extern_bcc/serial_bcc.cuh"
#include "extern_bcc/lch.cuh"

extern bool g_bcc_last_batch;
bool g_verbose = false;
int  local_block_size = 1000;
long maxThreadsPerBlock = 0;
static uint64_t pk(int a,int b){int l=std::min(a,b),h=std::max(a,b);
    return ((uint64_t)l<<32)|(uint32_t)h;}

int main(int argc,char**argv){
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p,0),"prop");
    maxThreadsPerBlock = p.maxThreadsPerBlock;
    int nb = atoi(argv[1]), n = atoi(argv[2]);
    std::vector<uint64_t> E;
    for (int i=3;i+1<argc;i+=2) E.push_back(pk(atoi(argv[i]),atoi(argv[i+1])));
    long m = (long)E.size();

    long rb=0, ri=0; long want = serial_bcc_count(E.data(), m, n, &rb, &ri);
    std::vector<int> rbeta; std::vector<char> rcut; std::vector<long> rsz;
    serial_bcc_partition(E.data(), m, n, rbeta, rcut, rsz);
    std::map<int,std::set<int>> truth;
    for (long e=0;e<m;++e){ truth[rbeta[e]].insert((int)(E[e]>>32));
                            truth[rbeta[e]].insert((int)(E[e]&0xFFFFFFFF)); }

    long bs = (m + nb - 1)/nb;
    GPU_BCG g(n, m, bs); g.numEdges = 0;
    for (long off=0; off<m; off+=bs) {
        long cur = std::min(bs, m-off);
        CUDA_CHECK(cudaMemcpy(g.d_edgelist, E.data()+off, cur*sizeof(uint64_t),
                              cudaMemcpyHostToDevice),"up");
        g_bcc_last_batch = (off + cur >= m);
        single_pass_batch_gpu(g, cur);
        printf("  after batch @%-3ld : n_s=%-5d n_b=%-5d |E|=%-5ld\n",
               off, g.n_s, g.n_b, (long)g.numEdges);
        if (getenv("BCC_SHOW")) {
            std::vector<uint64_t> hb(g.numEdges);
            if (g.numEdges) CUDA_CHECK(cudaMemcpy(hb.data(), g.updated_edgelist,
                g.numEdges*sizeof(uint64_t), cudaMemcpyDeviceToHost), "hb");
            std::vector<int> mb(n);
            CUDA_CHECK(cudaMemcpy(mb.data(), g.d_mapping, n*sizeof(int),
                cudaMemcpyDeviceToHost), "mb");
            printf("        E(H):");
            for (uint64_t e : hb) printf(" (%d,%d)", (int)(e>>32), (int)(e&0xFFFFFFFF));
            printf("\n        blocks:");
            for (int b2 = 0; b2 < g.n_b; ++b2) {
                printf(" %d<-{", b2);
                for (int v = 0; v < n; ++v) if (mb[v] == b2) printf("%d,", v);
                printf("}");
            }
            printf("\n");
            // BCC_NAMES: render E(H) in ORIGINAL vertex names, so a hand trace
            // can be checked batch by batch.  Every node prints its FULL
            // f^-1, because a node holding more than one vertex is exactly
            // what goes wrong when the filter is deferred:
            //   B<id>{...}  block,      O<id>{...}  own-node,
            //   a bare number for an own-node holding a single vertex,
            //   <id>{}      a node no original vertex maps to.
            if (getenv("BCC_NAMES")) {
                std::vector<std::string> nm(g.n_s);
                for (int x = 0; x < g.n_s; ++x) {
                    std::vector<int> res;
                    for (int v = 0; v < n; ++v) if (mb[v] == x) res.push_back(v);
                    bool blk = (x < g.n_b);
                    if (!blk && res.size() == 1) { nm[x] = std::to_string(res[0]); continue; }
                    std::string t = (blk ? "B" : "O") + std::to_string(x) + "{";
                    for (size_t k = 0; k < res.size(); ++k)
                        t += std::to_string(res[k]) + (k + 1 < res.size() ? "," : "");
                    nm[x] = t + "}";
                }
                printf("        H   :");
                for (uint64_t e : hb) {
                    int a = (int)(e >> 32), b3 = (int)(e & 0xFFFFFFFF);
                    printf(" %s--%s", a < g.n_s ? nm[a].c_str() : "??",
                                      b3 < g.n_s ? nm[b3].c_str() : "??");
                }
                printf("\n");
            }
        }
    }
    std::vector<int> M(n);
    CUDA_CHECK(cudaMemcpy(M.data(), g.d_mapping, n*sizeof(int), cudaMemcpyDeviceToHost),"M");
    std::vector<uint64_t> fin(g.numEdges);
    if (g.numEdges) CUDA_CHECK(cudaMemcpy(fin.data(), g.updated_edgelist,
        g.numEdges*sizeof(uint64_t), cudaMemcpyDeviceToHost),"fin");

    long br=0; for (uint64_t e: fin){int x=e>>32,y=e&0xFFFFFFFF;
        if (x>=g.n_b && y>=g.n_b) ++br; }
    long got = g.n_b + br;

    std::vector<int> lab, head;
    int ncomp = compute_lch(M, n, fin, g.n_s, g.n_b, lab, head);
    std::map<int,std::set<int>> cls;
    for (int v=0;v<n;++v) if (lab[v]>=0) cls[lab[v]].insert(v);
    std::set<std::set<int>> gotsets;
    for (auto& kv : cls) { std::set<int> s = kv.second;
        if (kv.first < (int)head.size() && head[kv.first] >= 0) s.insert(head[kv.first]);
        gotsets.insert(s); }
    std::set<std::set<int>> wantsets;
    for (auto& kv : truth) wantsets.insert(kv.second);

    if (getenv("BCC_SHOW")) {
        printf("\n  M[v] (= f(v)) per vertex, and n_b = %d  ->  id < %d is a BLOCK:\n   ",
               g.n_b, g.n_b);
        for (int v = 0; v < n; ++v) printf(" %d->%d%s", v, M[v], M[v] < g.n_b ? "(blk)" : "");
        printf("\n  so |f^-1(x)| per own-node id:\n   ");
        for (int x = g.n_b; x < g.n_s; ++x) {
            int c = 0, who = -1;
            for (int v = 0; v < n; ++v) if (M[v] == x) { ++c; who = v; }
            printf(" %d:{%s} ", x, c == 1 ? std::to_string(who).c_str() : (c ? "many" : ""));
        }
        printf("\n");
    }
    if (getenv("BCC_SHOW")) {
        printf("\n  OUR final H:  n_s=%d  n_b=%d  |E(H)|=%ld\n    V(H): ",
               g.n_s, g.n_b, (long)g.numEdges);
        for (int x = 0; x < g.n_s; ++x) printf(" %d%s", x, x < g.n_b ? "(blk)" : "");
        printf("\n    E(H): ");
        for (uint64_t e : fin) printf(" (%d,%d)", (int)(e>>32), (int)(e&0xFFFFFFFF));
        printf("\n");
    }
    printf("\n  f^-1 of each block id:\n");
    for (int b=0;b<g.n_b;++b){ printf("    block %d :", b); int c=0;
        for (int v=0;v<n;++v) if (M[v]==b) { printf(" %d",v); ++c; }
        printf("%s\n", c?"":"  <-- EMPTY (no residents)"); }
    if (getenv("BCC_SHOW")) {
        // THE ENUMERATION: bucket vertices by label, then add each head.
        printf("\n  --- BCC contents from (label[], head[]) ---\n");
        std::vector<std::vector<int>> bcc(ncomp);
        for (int v = 0; v < n; ++v)
            if (lab[v] >= 0) bcc[lab[v]].push_back(v);          // the class
        for (int j = 0; j < ncomp; ++j)
            if (head[j] >= 0) bcc[j].push_back(head[j]);        // + the head
        for (int j = 0; j < ncomp; ++j) {
            if (bcc[j].size() < 2) continue;                    // a lone vertex is no BCC
            std::sort(bcc[j].begin(), bcc[j].end());
            printf("    BCC %-3d :", j);
            for (int v : bcc[j]) printf(" %d", v);
            printf("      [class:");
            for (int v = 0; v < n; ++v) if (lab[v] == j) printf(" %d", v);
            printf("  head: %d]\n", head[j]);
        }
        printf("  --- which BCCs each vertex appears in ---\n");
        for (int v = 0; v < n; ++v) {
            std::vector<int> in;
            for (int j = 0; j < ncomp; ++j) {
                if (bcc[j].size() < 2) continue;
                if (std::find(bcc[j].begin(), bcc[j].end(), v) != bcc[j].end()) in.push_back(j);
            }
            if (in.empty()) continue;
            printf("    vertex %-2d : in %zu BCC%s ->", v, in.size(), in.size()>1?"s":" ");
            for (int j : in) printf(" %d", j);
            printf("%s\n", in.size() > 1 ? "   <== CUT VERTEX" : "");
        }
        printf("  --- label[] per vertex ---\n   ");
        for (int v = 0; v < n; ++v) printf(" %d:%d", v, lab[v]);
        printf("\n");
    }
    // BCC_KEY: everything an end-of-stream procedure can possibly see --
    // the final H's structure, which nodes are blocks, and every node's
    // |f^-1| -- plus the true answer. Two runs with the same KEY and a
    // different want= mean NO end-only procedure can be correct.
    if (getenv("BCC_KEY")) {
        printf("KEY n_s=%d n_b=%d nodes=", g.n_s, g.n_b);
        for (int x = 0; x < g.n_s; ++x) {
            int c = 0;
            for (int v = 0; v < n; ++v) if (M[v] == x) ++c;
            printf("%s%c%d", x ? "," : "", x < g.n_b ? 'B' : 'O', c);
        }
        std::vector<std::pair<int,int>> ke;
        for (uint64_t e : fin) {
            int a = (int)(e >> 32), b2 = (int)(e & 0xFFFFFFFF);
            ke.push_back({std::min(a,b2), std::max(a,b2)});
        }
        std::sort(ke.begin(), ke.end());
        printf(" edges=");
        for (size_t i = 0; i < ke.size(); ++i)
            printf("%s%d-%d", i ? "," : "", ke[i].first, ke[i].second);
        printf(" want=%ld\n", want);
    }
    printf("\n  count  got %ld  want %ld   %s\n", got, want, got==want?"OK":"MISMATCH");
    printf("  bcc sets %s  (ncomp=%d)\n", gotsets==wantsets?"MATCH":"MISMATCH", ncomp);
    if (gotsets!=wantsets) {
        printf("    want:"); for (auto&s:wantsets){printf(" {");for(int v:s)printf("%d,",v);printf("}");}
        printf("\n    got :"); for (auto&s:gotsets){printf(" {");for(int v:s)printf("%d,",v);printf("}");}
        printf("\n");
    }
    return 0;
}
