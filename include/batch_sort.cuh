/**
 * @file batch_sort.cuh
 * @brief This file contains the CPU and GPU algorithms for sorting multiple batches of size at most 1024.
 */

#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// d <= 1024
constexpr unsigned short MAX_BATCH_SIZE = 1024;

/**
 * @brief Kernel that sorts N batches of size d in parallel. Batch size d has to be a power of 2 because several merge steps with doubling length are performed.
 */
template <typename T>
__global__ void sortSmallBatch_k(T *batches, unsigned N, unsigned short d)
{

  //shared memory declaration
  __shared__ T shared_mem[2 * MAX_BATCH_SIZE];
  T *shared_in = shared_mem;
  T *shared_out = shared_mem + MAX_BATCH_SIZE;

  //load data from global memory to shared memory
  shared_in[threadIdx.x] = batches[blockDim.x * blockIdx.x + threadIdx.x];
  __syncthreads();

  //for loop on merge length
  for(unsigned short merge_lenght = 2; merge_lenght <= d; merge_lenght *= 2)
  {
    //define indexes
    unsigned batch_block_idx = threadIdx.x / merge_lenght; //Qt
    unsigned thread_local_idx = threadIdx.x - batch_block_idx * merge_lenght; //tidx

    //define shared memory pointers
    T *batch = shared_in + batch_block_idx * merge_lenght;
    unsigned local_A_size = merge_lenght / 2;
    unsigned local_B_size = merge_lenght / 2;

    T *A_local = batch;
    T *B_local = batch + local_A_size;
    
    // define diagonal endpoints
    uint2 K, P, Q;

    K.x = thread_local_idx <= local_A_size ? 0 : thread_local_idx - local_A_size;
    K.y = thread_local_idx <= local_A_size ? thread_local_idx : local_A_size;

    P.x = thread_local_idx <= local_B_size ? thread_local_idx : local_B_size;
    P.y = thread_local_idx <= local_B_size ? 0 : thread_local_idx - local_B_size;

    //binary search merge
    while (true)
    {
      unsigned offset = (K.y - P.y) / 2;
      Q.x = K.x + offset;
      Q.y = K.y - offset;
      if (Q.y == local_A_size || Q.x == 0 || A_local[Q.y] > B_local[Q.x - 1])
      {
        if (Q.x == local_B_size || Q.y == 0 || A_local[Q.y - 1] <= B_local[Q.x])
        {
          if (Q.y < local_A_size && (Q.x == local_B_size || A_local[Q.y] <= B_local[Q.x]))
          {
            shared_out[threadIdx.x] = A_local[Q.y];
          }
          else
          {
            shared_out[threadIdx.x] = B_local[Q.x];
          }
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

    //swap shared memory pointers
    T *temp = shared_in;
    shared_in = shared_out;
    shared_out = temp;
    __syncthreads();
  }

  //write data back to global memory
  batches[blockDim.x * blockIdx.x + threadIdx.x] = shared_in[threadIdx.x];

  return;
}

/**
 * @brief Host function to check correctness of the GPU batch sort kernel.
 */
template <class T>
float sort_batch_cpu(std::vector<T> &batches, unsigned N, unsigned short d)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(unsigned batch_idx = 0; batch_idx < N; batch_idx++)
  {
    auto curr_batch_start = batches.begin() + batch_idx * d;
    std::sort(curr_batch_start, curr_batch_start + d);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}
