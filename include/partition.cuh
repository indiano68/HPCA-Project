/**
 * @file partition.cuh
 * @brief This file contains the kernels and device functions to perform merge path partitioning, both coarse (partition_k) and fine (fine_partition).
 */

#pragma once

//partition_k
constexpr unsigned THREADS_PER_BLK_PARTITION = 32;

/**
 * @brief Device function that finds the intersection of the merge path with a diagonal identified by the endpoints K and P.
 * It is used both in coarse partitioning (partition_k, happens in global memory) and fine partitioning (fine_partition, happens in shared memory).
 */
template <class T>
__device__ __forceinline__ int2 explorative_search(const T *A_ptr, const size_t A_size, 
                                                   const T *B_ptr, const size_t B_size, 
                                                   int2 K, int2 P)
{

  while (true)
  {
    uint32_t offset = (K.y - P.y) / 2;
    int2 Q = {K.x + (int)offset, K.y - (int)offset};

    if (Q.y == A_size || Q.x == 0 || A_ptr[Q.y] > B_ptr[Q.x - 1])
    {
      if (Q.x == B_size || Q.y == 0 || A_ptr[Q.y - 1] <= B_ptr[Q.x])
      {
        return Q;
      }
      else
      {
        K = {Q.x + 1, Q.y - 1};
      }
    }
    else
    {
      P = {Q.x - 1, Q.y + 1};
    }
  }

}

/**
 * @brief Kernel that performs coarse partitioning step. Writes the partitioning points to global memory (Q_global).
 */
template <class T>
__global__ void partition_k(const T *A_ptr, const size_t A_size,
                            const T *B_ptr, const size_t B_size,
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

    return;
  }
}

/*
 * @brief Device function that performs fine partitioning in shared memory. Used by mergeLarge_tiled_k.
*/
template <class T>
__device__ int2 fine_partition(const T *A_box, const size_t A_size,
                              const T *B_box, const size_t B_size,
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


