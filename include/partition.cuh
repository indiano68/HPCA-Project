#pragma once
#include <config.h> 

template <class T>
__global__ void partitioner(const T *v_1_ptr,
                            const int v_1_size,
                            const T *v_2_ptr,
                            const int v_2_size,
                            int2 *v_Q_ptr,
                            const int v_Q_size)
{
    int thread_idx = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    coordinate K = {0, 0};
    coordinate P = {0, 0};
    coordinate Q = {0, 0};
    if (thread_idx > v_1_size)
    {
        K.x = thread_idx - v_1_size;
        K.y = v_1_size;
        P.x = v_1_size;
        P.y = thread_idx - v_1_size;
    }
    else
    {
        K.y = thread_idx;
        P.x = thread_idx;
    }
    while (true && thread_idx < v_1_size +v_2_size && threadIdx.x == 0)
    {
        size_t offset = abs((K.y - P.y)) / 2;
        Q.x = K.x + offset;
        Q.y = K.y - offset;
        if (Q.y >= 0 && Q.x <= v_2_size && (Q.y == v_1_size || Q.x == 0 || v_1_ptr[Q.y] > v_2_ptr[Q.x - 1]))
        {
            if (Q.x == v_2_size || Q.y == 0 || v_1_ptr[Q.y - 1] <= v_2_ptr[Q.x])
            {
                v_Q_ptr[blockIdx.x] = Q;
                break;
            }
            else
            {
                K.x = Q.x + 1;
                K.y = Q.y - 1;
            }
        }
        else
        {
            P.x = Q.x - 1;
            P.y = Q.y + 1;
        }
    }
}

template <class T>
__global__ void partition_k_gpu_packed(const T *A_ptr,
                                       const size_t A_size,
                                       const T *B_ptr,
                                       const size_t B_size,
                                       int2 *Q_global, size_t n_partitions = TILE_SIZE * TILES_PER_BLOCK)
{

  unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  unsigned diag = min((unsigned)n_partitions * (tid + 1), (unsigned)(A_size + B_size));

  unsigned Q_global_size = (A_size + B_size + n_partitions - 1) / n_partitions;

  if(tid < Q_global_size)
  {

    int2 K_explorative = {0, 0}, P_explorative = {0, 0};

    K_explorative.x = (diag > A_size) ? diag - A_size : 0;
    K_explorative.y = (diag < A_size) ? diag : A_size;
    P_explorative.x = (diag < B_size) ? diag : B_size;
    P_explorative.y = (diag > B_size) ? diag - B_size : 0;

    Q_global[tid] = explorative_search(A_ptr, A_size, B_ptr, B_size, K_explorative, P_explorative);

    if constexpr(DEBUG) printf("Block %d thread %d diag = %d, K(%d,%d) - P(%d,%d), found Q(%d,%d)\n", blockIdx.x, threadIdx.x, diag, K_explorative.x, K_explorative.y, P_explorative.x, P_explorative.y, Q_global[tid].x, Q_global[tid].y);

    return;
  }
}
