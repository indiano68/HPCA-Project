#pragma once
#include <vector>
#include <diag_search.cuh>

//#define THREADS_PER_BLOCK 128

int constexpr THREADS_PER_BLOCK = 1024;

template <class T>
__global__ void mergeSmall_k_gpu_seq(const T *A_ptr,
                                     const size_t A_size,
                                     const T *B_ptr,
                                     const size_t B_size,
                                     T *M_ptr,
                                     const size_t M_size)
{
  static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");

  size_t i = 0, j = 0;

  while (i + j < M_size)
  {
    if (i >= A_size)
    {
      M_ptr[i + j] = B_ptr[j];
      j++;
    }
    else if (j >= B_size || A_ptr[i] < B_ptr[j])
    {
      M_ptr[i + j] = A_ptr[i];
      i++;
    }
    else
    {
      M_ptr[i + j] = B_ptr[j];
      j++;
    }
  }
}

template <class T>
__global__ void mergeSmall_k_gpu(const T *A_ptr,
                                 const size_t A_size,
                                 const T *B_ptr,
                                 const size_t B_size,
                                 T *M_ptr,
                                 const size_t M_size)
{
  static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");

  int tid = threadIdx.x;
  int2 K = {0, 0}, P = {0, 0};

  if (tid > A_size)
  {
    K.x = tid - A_size, K.y = A_size;
    P.x = A_size, P.y = tid - A_size;
  }
  else
  {
    K.y = tid, P.x = tid;
  }

  while (true)
  {
    uint32_t offset = abs(K.y - P.y) / 2;
    int2 Q = {K.x + (int)offset, K.y - (int)offset};

    // i primi due if detectano se il punto è plausibile
    if (Q.y >= 0 && Q.x <= B_size && (Q.y == A_size || Q.x == 0 || A_ptr[Q.y] >= B_ptr[Q.x - 1]))
    {
      if (Q.x == B_size || Q.y == 0 || A_ptr[Q.y - 1] <= B_ptr[Q.x])
      {
        if (Q.y < A_size && (Q.x == B_size || A_ptr[Q.y] <= B_ptr[Q.x])) // traccia la bacchetta per scegliaere se inserire l'elemento di A o di B
        {
          M_ptr[tid] = A_ptr[Q.y];
        }
        else
        {
          M_ptr[tid] = B_ptr[Q.x];
        }
        break;
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

template <class T>
__global__ void merge_k_gpu_triangles(const T *A_ptr,
                                            const size_t A_size,
                                            const T *B_ptr,
                                            const size_t B_size,
                                            T *M_ptr)
{

  //static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");

  __shared__ T A_shared[THREADS_PER_BLOCK + 1], B_shared[THREADS_PER_BLOCK + 1];
  __shared__ int2 Q_base;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  int last_tid_of_block = blockDim.x * (blockIdx.x + 1) - 1;

  if(tid == last_tid_of_block)
  {
    int2 K_explorative = {0, 0}, P_explorative = {0, 0};

    K_explorative.x = (tid > A_size) ? tid - A_size : 0;
    K_explorative.y = (tid < A_size) ? tid : A_size;
    P_explorative.x = (tid < B_size) ? tid : B_size;
    P_explorative.y = (tid > B_size) ? tid - B_size : 0;

    Q_base = explorative_search(A_ptr, A_size, B_ptr, B_size, K_explorative, P_explorative);

    if(DEBUG) printf("Explorative search (block %d) found Q_base(%d,%d)\n", blockIdx.x, Q_base.x, Q_base.y);
  }
  __syncthreads();

  int x_end = Q_base.x;
  int y_end = Q_base.y;

  int x_start = x_end >= blockDim.x ? x_end - blockDim.x + 1: 0;
  int y_start = y_end >= blockDim.x ? y_end - blockDim.x + 1: 0;

  bool blk_left_border = (x_start == 0);
  bool blk_top_border = (y_start == 0);
  bool blk_bottom_border = (y_end == A_size);
  bool blk_right_border = (x_end == B_size);

  A_shared[threadIdx.x + 1] = y_start + threadIdx.x < A_size ? A_ptr[y_start + threadIdx.x] : 0;
  B_shared[threadIdx.x + 1] = x_start + threadIdx.x < B_size ? B_ptr[x_start + threadIdx.x] : 0;

  if(threadIdx.x == 0)
  {
    A_shared[0] = (y_start > 0) ? A_ptr[y_start - 1] : 0;
    B_shared[0] = (x_start > 0) ? B_ptr[x_start - 1] : 0;
  }

  int base = x_end - x_start; //base del triangolo
  int height = y_end - y_start; //altezza del triangolo

  if(DEBUG && threadIdx.x == 0) printf("Block %d, base = %d, height = %d\n", blockIdx.x, base, height);

  //complementare di threadIdx.x nel blocco
  int reverse_tid = blockDim.x - threadIdx.x - 1;

  int2 K, P;

  K.x = base >= reverse_tid ? base - reverse_tid : 0;
  K.y = base >= reverse_tid ? height : height + base - reverse_tid;

  P.x = height >= reverse_tid ? base : base + height - reverse_tid;
  P.y = height >= reverse_tid ? height - reverse_tid : 0;

  if(DEBUG) printf("Block %d thread %d search range: K(%d,%d) P(%d,%d)\n", blockIdx.x, threadIdx.x, K.x, K.y, P.x, P.y);

  __syncthreads();

  block_bin_search(A_shared + 1, B_shared + 1, K, P, M_ptr, blk_left_border, blk_right_border, blk_top_border, blk_bottom_border, base, height);

  if(DEBUG) printf("Block %d thread %d finished binsearch\n", blockIdx.x, threadIdx.x);

  return;
}

template <class T>
__global__ void partition_k_gpu(const T *A_ptr,
                              const size_t A_size,
                              const T *B_ptr,
                              const size_t B_size,
                              int2 *Q_global)
{

  //this kernel is launched with one thread per block (for now), THREADS_PER_BLOCK refers to the number of threads per block to be used in the merge stage

  int tid = THREADS_PER_BLOCK * (blockIdx.x + 1) - 1;

  int2 K_explorative = {0, 0}, P_explorative = {0, 0};

  K_explorative.x = (tid > A_size) ? tid - A_size : 0;
  K_explorative.y = (tid < A_size) ? tid : A_size;
  P_explorative.x = (tid < B_size) ? tid : B_size;
  P_explorative.y = (tid > B_size) ? tid - B_size : 0;

  Q_global[blockIdx.x] = explorative_search(A_ptr, A_size, B_ptr, B_size, K_explorative, P_explorative);

  return;
}

template <class T>
__global__ void merge_k_gpu_squares(const T *A_ptr, 
                                    const size_t A_size, 
                                    const T *B_ptr, 
                                    const size_t B_size, 
                                    T *M_ptr, 
                                    const int2 *Q_global)
{

  __shared__ T A_shared[THREADS_PER_BLOCK], B_shared[THREADS_PER_BLOCK];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  int2 Q_prev_blk = blockIdx.x == 0 ? (int2){0, 0} : Q_global[blockIdx.x - 1];
  int2 Q_curr_blk = Q_global[blockIdx.x];

  int base = Q_curr_blk.x - Q_prev_blk.x;
  int height = Q_curr_blk.y - Q_prev_blk.y;

  A_shared[threadIdx.x] = (Q.prev_blk.x == 0 || Q.prev_blk.x + threadIdx.x - 1 >= A_size) ? 0 : A_ptr[Q_prev_blk.x + threadIdx.x - 1];
  B_shared[threadIdx.x] = (Q.prev_blk.y == 0 || Q.prev_blk.y + threadIdx.x - 1 >= B_size) ? 0 : B_ptr[Q_prev_blk.y + threadIdx.x - 1];

  __syncthreads();

  int2 K, P;

  

  return;
}

template <class T>
__global__ void mergeSmall_k_gpu_claudio(const T *A_ptr,
                                    const size_t A_size,
                                    const T *B_ptr,
                                    const size_t B_size,
                                    T *M_ptr,
                                    const size_t M_size)
{
  static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");

  if(threadIdx.x == 0 && A_size > B_size)
  {
    printf("A size is greater than B size, exploding...\n");
    return;
  }

  int tid = threadIdx.x;
  int2 K = {0, 0}, P = {0, 0};

  K.x = (tid > (int)A_size) ? (tid - (int)A_size) : 0;
  K.y = (tid < (int)A_size) ? tid : (int)A_size;
  P.x = (tid < (int)B_size) ? tid : (int)B_size;
  P.y = (tid > (int)B_size) ? (tid - (int)B_size) : 0;

  while (true)
  {
    uint32_t offset = abs(K.y - P.y) / 2;
    int2 Q = {K.x + (int)offset, K.y - (int)offset};

    bool condition1 = (Q.y == A_size || Q.x == 0 || A_ptr[Q.y] >= B_ptr[Q.x - 1]);
    bool condition2 = (Q.x == B_size || Q.y == 0 || A_ptr[Q.y - 1] <= B_ptr[Q.x]);

    if (condition1 && condition2)
    {
      M_ptr[tid] = (Q.x == B_size || A_ptr[Q.y] <= B_ptr[Q.x]) ? A_ptr[Q.y] : B_ptr[Q.x];
      break;
    }

    condition1 ? K = {Q.x + 1, Q.y - 1} : P = {Q.x - 1, Q.y + 1};
  }
}