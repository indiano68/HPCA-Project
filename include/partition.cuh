#pragma once
#include <config.h> 

template <class T>
__global__ void partition_k(const T *A_ptr,
                                       const size_t A_size,
                                       const T *B_ptr,
                                       const size_t B_size,
                                       int2 *Q_global, size_t partition_size)
{

  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned diag = min((unsigned)partition_size * (tid + 1), (unsigned)(A_size + B_size));

  unsigned Q_global_size = (A_size + B_size + partition_size - 1) / partition_size;

  if(tid < Q_global_size)
  {

    int2 K_explorative = {0, 0}, P_explorative = {0, 0};

    K_explorative.x = (diag > A_size) ? diag - A_size : 0;
    K_explorative.y = (diag < A_size) ? diag : A_size;
    P_explorative.x = (diag < B_size) ? diag : B_size;
    P_explorative.y = (diag > B_size) ? diag - B_size : 0;

    Q_global[tid] = explorative_search(A_ptr, A_size, B_ptr, B_size, K_explorative, P_explorative);

    if constexpr(DEBUG) printf("PARTITION_PACKED: Block %d thread %d diag = %d, K(%d,%d) - P(%d,%d), found Q(%d,%d)\n", blockIdx.x, threadIdx.x, diag, K_explorative.x, K_explorative.y, P_explorative.x, P_explorative.y, Q_global[tid].x, Q_global[tid].y);

    return;
  }
}

template <class T>
__device__ int2 partition_box(const T *A_box,
                              const size_t A_size,
                              const T *B_box,
                              const size_t B_size,
                              size_t partition_size)
{

  unsigned diag = min((unsigned)partition_size * threadIdx.x, (unsigned)(A_size + B_size));

  int2 K_explorative, P_explorative;

  K_explorative.x = (diag > A_size) ? diag - A_size : 0;
  K_explorative.y = (diag < A_size) ? diag : A_size;
  P_explorative.x = (diag < B_size) ? diag : B_size;
  P_explorative.y = (diag > B_size) ? diag - B_size : 0;

  return explorative_search(A_box, A_size, B_box, B_size, K_explorative, P_explorative);
}


