// Cost of compressing every BCC (BCC_MIN_EDGES=1) vs only real blocks (=2).
// Reports the id budget and the carried edge set, which is what the two rules
// actually trade against each other.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <functional>
#include <vector>
#include <cuda_runtime.h>
#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"
#include "extern_bcc/single_pass.cuh"

bool g_verbose = false;
int  local_block_size = 1000;
long maxThreadsPerBlock = 0;

static uint64_t pk(int a,int b){int l=std::min(a,b),h=std::max(a,b);
    return ((uint64_t)l<<32)|(uint32_t)h;}

int main(int argc,char**argv){
    int  n = (argc>1)?atoi(argv[1]):200000;
    long m = (argc>2)?atol(argv[2]):600000;
    int  k = (argc>3)?atoi(argv[3]):8;
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p,0),"prop");
    maxThreadsPerBlock = p.maxThreadsPerBlock;

    std::mt19937 rng(12345);
    std::set<uint64_t> uniq;
    while ((long)uniq.size() < m) { int u=rng()%n, v=rng()%n; if(u!=v) uniq.insert(pk(u,v)); }
    std::vector<uint64_t> E(uniq.begin(), uniq.end());
    m = (long)E.size();
    std::shuffle(E.begin(), E.end(), rng);

    long bs = (m + k - 1)/k;
    GPU_BCG g(n, m, bs);
    g.numEdges = 0;

    long peakE = 0, sumE = 0; int peak_ns = 0, peak_nb = 0;
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (long off=0; off<m; off+=bs) {
        long cur = std::min(bs, m-off);
        CUDA_CHECK(cudaMemcpy(g.d_edgelist, E.data()+off, cur*sizeof(uint64_t),
                              cudaMemcpyHostToDevice),"up");
        single_pass_batch_gpu(g, cur);
        // Is the CARRIED H a forest? Necessary condition |E| <= n_s - 1, then
        // an actual acyclicity test by union-find over the edge list.
        {
            std::vector<uint64_t> h(g.numEdges);
            if (g.numEdges)
                CUDA_CHECK(cudaMemcpy(h.data(), g.updated_edgelist,
                           g.numEdges*sizeof(uint64_t), cudaMemcpyDeviceToHost), "h");
            std::vector<int> par(g.n_s); for (int i=0;i<g.n_s;++i) par[i]=i;
            std::function<int(int)> f=[&](int x){ while(par[x]!=x){par[x]=par[par[x]];x=par[x];} return x; };
            long cyc = 0;
            for (uint64_t e : h) {
                int a=(int)(e>>32), b=(int)(e&0xFFFFFFFF);
                if (a>=g.n_s||b>=g.n_s) continue;
                int ra=f(a), rb=f(b);
                if (ra==rb) ++cyc; else par[ra]=rb;
            }
            printf("    batch: n_s=%-8d |E|=%-8ld  |E|<=n_s-1 %s   cycle-edges=%ld %s\n",
                   g.n_s, (long)g.numEdges, g.numEdges <= g.n_s-1 ? "YES":"NO ",
                   cyc, cyc==0 ? "(FOREST)" : "(NOT a forest)");
        }
        peakE = std::max(peakE, g.numEdges); sumE += g.numEdges;
        peak_ns = std::max(peak_ns, g.n_s); peak_nb = std::max(peak_nb, g.n_b);
    }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);

    const char* e = getenv("BCC_MIN_EDGES");
    printf("min_edges=%-2s n=%d m=%ld k=%d | peak n_b=%-8d peak n_s=%-8d "
           "peak |E|=%-9ld sum |E|=%-10ld final n_b=%-8d final |E|=%-9ld %7.1f ms\n",
           (e&&atoi(e)==1)?"1":"2", n, m, k, peak_nb, peak_ns, peakE, sumE,
           g.n_b, g.numEdges, ms);
    return 0;
}
