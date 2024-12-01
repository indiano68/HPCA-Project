#pragma once

template <typename T>
__global__ void mergeSmallBatch_k(T *batches, unsigned short *A_sizes, unsigned N, unsigned short d)
{
  __shared__ T shared_in[1024];
  __shared__ T shared_out[1024];

  //define indexes
  unsigned batch_per_block = blockDim.x / d;
  unsigned batch_block_idx = threadIdx.x / d;
  unsigned thread_local_idx = threadIdx.x - batch_block_idx * d;
  unsigned thread_global_idx = batch_block_idx + blockIdx.x * batch_per_block;

  //load data into shared memory
  if(thread_global_idx < N)
  {
    shared_in[threadIdx.x] = batches[blockDim.x * blockIdx.x + threadIdx.x];
  }
  __syncthreads();

  //parallel merge
  if(thread_global_idx < N)
  {

    T *batch = shared_in + batch_block_idx * d;
    unsigned local_A_size = A_sizes[thread_global_idx];
    unsigned local_B_size = d - local_A_size;

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
  if(thread_global_idx < N)
  {
    batches[blockDim.x * blockIdx.x + threadIdx.x] = shared_out[threadIdx.x];
  }

  return;
}