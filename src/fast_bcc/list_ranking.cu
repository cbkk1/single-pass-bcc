#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#include "utility/cuda_utility.cuh"
#include "fast_bcc/list_ranking.cuh"

__global__
void initRankNext(unsigned long long *devRankNext, const int *devNext, int N) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < N) {
        devRankNext[thid] = (((unsigned long long)0) << 32) + devNext[thid] + 1;
    }
}

__global__ 
void updateRankNext(int loopsWithoutSync, unsigned long long *devRankNext, int *devNotAllDone, int N) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thid < N) {

        unsigned long long rankNext = devRankNext[thid];
        for (int i = 0; i < loopsWithoutSync; i++) {
            // The flag is cleared by the HOST before each launch, never here.
            // Clearing it in-kernel from thread 0 races with the stores below:
            // there is no synchronization inside this loop, so thread 0's clear
            // at its iteration i can land AFTER another thread's store of 1 at
            // its own i == loopsWithoutSync-1. The host then leaves the
            // convergence loop with the ranks only partially jumped, and every
            // consumer of devRank (find_parent, compute_level, first/last_occ)
            // silently reads a non-ranking.
            int rank = rankNext >> 32;
            int nxt = rankNext - 1;

            if (nxt != -1) {
                unsigned long long grandNext = devRankNext[nxt];

                rank += (grandNext >> 32) + 1;
                nxt = grandNext - 1;

                rankNext = (((unsigned long long)rank) << 32) + nxt + 1;
                atomicExch(devRankNext + thid, rankNext);

                if (i == loopsWithoutSync - 1) {
                    *devNotAllDone = 1;
                }
            }
        }
    }
}

__global__ 
void extractHighBits(const unsigned long long *devRankNext, int *devRank, int N) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < N) {
        devRank[thid] = devRankNext[thid] >> 32;  // Shift right by 32 bits to get the high part
    }
}

// param: int *devNext is the input array 
// param: int *devRank is the output array 
// param: int N is the size of input array
void CudaSimpleListRank(int *devRank, int N, int *devNext, 
    int *notAllDone, ull *devRankNext, int *devNotAllDone) {

    int threadsPerBlock = 1024; // This can be tuned based on your device capabilities
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize devRankNext with initRankNext kernel
    initRankNext<<<numBlocks, threadsPerBlock>>>(devRankNext, devNext, N);
    CUCHECK(cudaDeviceSynchronize());
    const int loopsWithoutSync = 5;

    do {
        CUCHECK(cudaMemset(devNotAllDone, 0, sizeof(int)));
        updateRankNext<<<numBlocks, threadsPerBlock>>>(loopsWithoutSync, devRankNext, devNotAllDone, N);
        CUCHECK(cudaDeviceSynchronize());
        CUCHECK(cudaMemcpy(notAllDone, devNotAllDone, sizeof(int), cudaMemcpyDeviceToHost));
    } while (*notAllDone);

    // Launch the kernel
    extractHighBits<<<numBlocks, threadsPerBlock>>>(devRankNext, devRank, N);
    CUCHECK(cudaDeviceSynchronize());
}