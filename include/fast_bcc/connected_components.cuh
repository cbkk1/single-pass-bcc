#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

#include <string>
#include <iostream>
#include <cuda_runtime.h>

#include "extern_bcc/extern_bcc.cuh"       
#include "extern_bcc/bcg_memory_utils.cuh"

// nb_keep: the block watermark for step 4.  Ids below it are carried block
// nodes; a tree edge whose PARENT is one is never fenced.  0 disables it.
void CC(GPU_BCG& g_bcg_ds, const bool isLastBatch, const int nb_keep = 0);

// The step-4 form in force (4 = block-aware default, 0 = stock + filter).
int bcc_fence_mode_resolved();

#endif //CONNECTED_COMPONENTS_H
