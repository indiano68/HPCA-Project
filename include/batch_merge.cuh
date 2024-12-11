/**
 * @file batch_merge.cuh
 * @brief This file contains cpu and gpu algorithms for batch merging, as weel as a helper function to generate the random batches on the CPU.
 * It is used in the main_batch.cu file.
 */

#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

// d <= 1024
static constexpr unsigned short MAX_BATCH_SIZE = 1024;

using std::numeric_limits;

template <typename T>
__global__ void mergeSmallBatch_k(T *batches, unsigned short *A_sizes, unsigned N, unsigned short d)
{
  // since MAX_BATCH_SIZE is 1024, we can use shared memory to store both input and output
  __shared__ T shared_in[MAX_BATCH_SIZE];
  __shared__ T shared_out[MAX_BATCH_SIZE];

  // define indexes
  unsigned batch_per_block = blockDim.x / d;
  unsigned batch_block_idx = threadIdx.x / d;                                  // Qt
  unsigned thread_local_idx = threadIdx.x - batch_block_idx * d;               // tidx
  unsigned thread_global_idx = batch_block_idx + blockIdx.x * batch_per_block; // gbx

  // load data into shared memory
  if (thread_global_idx < N)
  {
    shared_in[threadIdx.x] = batches[blockDim.x * blockIdx.x + threadIdx.x];
  }
  __syncthreads();

  // parallel merge
  if (thread_global_idx < N)
  {
    
    // define shared memory pointers and retrieve local A size from global memory
    T *batch = shared_in + batch_block_idx * d;
    unsigned local_A_size = A_sizes[thread_global_idx];
    unsigned local_B_size = d - local_A_size;

    T *A_local = batch;
    T *B_local = batch + local_A_size;

    // define diagonal endpoints
    uint2 K, P, Q;

    K.x = thread_local_idx <= local_A_size ? 0 : thread_local_idx - local_A_size;
    K.y = thread_local_idx <= local_A_size ? thread_local_idx : local_A_size;

    P.x = thread_local_idx <= local_B_size ? thread_local_idx : local_B_size;
    P.y = thread_local_idx <= local_B_size ? 0 : thread_local_idx - local_B_size;

    // binary search merge
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
  }
  __syncthreads();

  // write data back to global memory if the thread is within the bounds
  if (thread_global_idx < N)
  {
    batches[blockDim.x * blockIdx.x + threadIdx.x] = shared_out[threadIdx.x];
  }

  return;
}

/**
 * @brief Generate N batches of size d (|A_i| + |B_i| = d) and sort them.
 * @returns A vector of offsets with the split points between A and B for each batch.
 */
template <class T>
const std::vector<unsigned short> build_and_sort_batches(std::vector<T> &batches, unsigned N, unsigned short d, T min = numeric_limits<T>::min(), T max = numeric_limits<T>::max())
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min, max);
  std::generate(batches.begin(), batches.end(), [&dis, &gen]()
                { return static_cast<T>(dis(gen)); });

  std::uniform_int_distribution<unsigned short> splitter(1, d - 1);
  std::vector<unsigned short> offsets(N);
  std::generate(offsets.begin(), offsets.end(), [&splitter, &gen]()
                { return splitter(gen); });

  #pragma omp parallel for
  for (unsigned batch_idx = 0; batch_idx < N; batch_idx++)
  {
    auto curr_batch_start = batches.begin() + batch_idx * d;
    std::sort(curr_batch_start, curr_batch_start + offsets[batch_idx]);
    std::sort(curr_batch_start + offsets[batch_idx], curr_batch_start + d);
  }
  return offsets;
}

/**
 * @brief Merge N batches of size d (|A_i| + |B_i| = d) on the CPU.
 */
template <class T>
float merge_batch_cpu(std::vector<T> &batches, const std::vector<unsigned short> &offsets, unsigned N, unsigned short d)
{
  auto start = std::chrono::high_resolution_clock::now();
  for (unsigned batch_idx = 0; batch_idx < N; batch_idx++)
  {
    auto curr_batch_start = batches.begin() + batch_idx * d;
    std::inplace_merge(curr_batch_start, curr_batch_start + offsets[batch_idx], curr_batch_start + d);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}