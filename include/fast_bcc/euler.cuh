#ifndef EULER_TOUR_CUH
#define EULER_TOUR_CUH

#include "utility/cuda_utility.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"

// `c` = number of trees in the spanning forest (1 for a connected graph).
void cuda_euler_tour(int N, int c, GPU_BCG& g_bcg_ds);

#endif // EULER_TOUR_CUH