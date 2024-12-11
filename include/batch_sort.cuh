#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// d <= 1024
constexpr unsigned short MAX_BATCH_SIZE = 1024;

//works only id d is a power of 2
template <typename T>
__global__ void mergeSmallBatch_for_k(T *batches, unsigned N, unsigned short d)
{
  __shared__ T shared_in[MAX_BATCH_SIZE];
  __shared__ T shared_out[MAX_BATCH_SIZE];

  //define indexes
  unsigned batch_block_idx = threadIdx.x / d;
  unsigned thread_local_idx = threadIdx.x - batch_block_idx * d;

  //load data into shared memory
  // if(thread_global_idx < N)
  {
    shared_in[threadIdx.x] = batches[blockDim.x * blockIdx.x + threadIdx.x];
  }
  __syncthreads();

  //parallel merge
  // if(thread_global_idx < N)
  {

    T *batch = shared_in + batch_block_idx * d;
    unsigned local_A_size = d / 2;
    unsigned local_B_size = d / 2;

    T *A_local = batch;
    T *B_local = batch + local_A_size;
    
    uint2 K, P, Q;

    K.x = thread_local_idx <= local_A_size ? 0 : thread_local_idx - local_A_size;
    K.y = thread_local_idx <= local_A_size ? thread_local_idx : local_A_size;

    P.x = thread_local_idx <= local_B_size ? thread_local_idx : local_B_size;
    P.y = thread_local_idx <= local_B_size ? 0 : thread_local_idx - local_B_size;

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

  //write data back to global memory
  // if(thread_global_idx < N)
  {
    batches[blockDim.x * blockIdx.x + threadIdx.x] = shared_out[threadIdx.x];
  }

  return;
}

//works only id d is a power of 2
template <typename T>
__global__ void sortSmallBatch_k(T *batches, unsigned N, unsigned short d)
{

  __shared__ T shared_mem[2 * MAX_BATCH_SIZE];
  T *shared_in = shared_mem;
  T *shared_out = shared_mem + MAX_BATCH_SIZE;

  //unsigned global_idx = threadIdx.x / d + blockIdx.x * (blockDim.x / d);

  //load data into shared memory
  // if(global_idx < N)
  {
    shared_in[threadIdx.x] = batches[blockDim.x * blockIdx.x + threadIdx.x];
  }
  __syncthreads();

  // if(global_idx < N)
  {
    for(unsigned short merge_lenght = 2; merge_lenght <= d; merge_lenght *= 2)
    {
      //define indexes
      unsigned batch_block_idx = threadIdx.x / merge_lenght;
      unsigned thread_local_idx = threadIdx.x - batch_block_idx * merge_lenght;

      T *batch = shared_in + batch_block_idx * merge_lenght;
      unsigned local_A_size = merge_lenght / 2;
      unsigned local_B_size = merge_lenght / 2;

      T *A_local = batch;
      T *B_local = batch + local_A_size;
      
      uint2 K, P, Q;

      K.x = thread_local_idx <= local_A_size ? 0 : thread_local_idx - local_A_size;
      K.y = thread_local_idx <= local_A_size ? thread_local_idx : local_A_size;

      P.x = thread_local_idx <= local_B_size ? thread_local_idx : local_B_size;
      P.y = thread_local_idx <= local_B_size ? 0 : thread_local_idx - local_B_size;

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
  }

  //write data back to global memory 
  // if(global_idx < N)
  {
    batches[blockDim.x * blockIdx.x + threadIdx.x] = shared_in[threadIdx.x];
  }

  return;
}

template <class T>
void sort_batch_cpu(std::vector<T> &batches, unsigned N, unsigned short d)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(unsigned batch_idx = 0; batch_idx < N; batch_idx++)
  {
    auto curr_batch_start = batches.begin() + batch_idx * d;
    std::sort(curr_batch_start, curr_batch_start + d);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "CPU sort time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}
