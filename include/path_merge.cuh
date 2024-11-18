#pragma once
#include <vector>


typedef int2 coordinate;
int constexpr THREADS_PER_BLOCK = 1024;
int constexpr DEBUG = false;


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
__global__ void mergeSmall_k(const T *v_1_ptr,
                             const size_t v_1_size,
                             const T *v_2_ptr,
                             const size_t v_2_size,
                             T *v_out_ptr,
                             const size_t v_out_size)
{
    int thread_idx = threadIdx.x;
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

    while (true && thread_idx < v_out_size)
    {
        size_t offset = abs((K.y - P.y)) / 2;
        Q.x = K.x + offset;
        Q.y = K.y - offset;
        if (Q.y >= 0 && Q.x <= v_2_size && (Q.y == v_1_size || Q.x == 0 || v_1_ptr[Q.y] > v_2_ptr[Q.x - 1]))
        {
            if (Q.x == v_2_size || Q.y == 0 || v_1_ptr[Q.y - 1] <= v_2_ptr[Q.x])
            {
                if (Q.y < v_1_size && (Q.x == v_2_size || v_1_ptr[Q.y] <= v_2_ptr[Q.x]))
                {
                    v_out_ptr[thread_idx] = v_1_ptr[Q.y];
                }
                else
                {
                    v_out_ptr[thread_idx] = v_2_ptr[Q.x];
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

template <class T>
__global__ void merge_k_naive(const T *v_1_ptr,
                              const size_t v_1_size,
                              const T *v_2_ptr,
                              const size_t v_2_size,
                              T *v_out_ptr,
                              const size_t v_out_size)
{
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
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

    while (true && thread_idx < v_out_size)
    {
        size_t offset = abs((K.y - P.y)) / 2;
        Q.x = K.x + offset;
        Q.y = K.y - offset;
        if (Q.y >= 0 && Q.x <= v_2_size && (Q.y == v_1_size || Q.x == 0 || v_1_ptr[Q.y] > v_2_ptr[Q.x - 1]))
        {
            if (Q.x == v_2_size || Q.y == 0 || v_1_ptr[Q.y - 1] <= v_2_ptr[Q.x])
            {
                if (Q.y < v_1_size && (Q.x == v_2_size || v_1_ptr[Q.y] <= v_2_ptr[Q.x]))
                {
                    v_out_ptr[thread_idx] = v_1_ptr[Q.y];
                }
                else
                {
                    v_out_ptr[thread_idx] = v_2_ptr[Q.x];
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


template <class T>
__device__ inline int2 explorative_search(const T *A_ptr, const size_t A_size, const T *B_ptr, const size_t B_size, int2 K, int2 P)
{

while (true)
  {
    uint32_t offset = abs(K.y - P.y) / 2;
    int2 Q = {K.x + (int)offset, K.y - (int)offset};

    // i primi due if detectano se il punto è plausibile
    if (Q.y == A_size || Q.x == 0 || A_ptr[Q.y] >= B_ptr[Q.x - 1])
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

template <class T>
__device__ void block_bin_search(const T *A_local, const T *B_local, int2 K, int2 P, T *M_global, bool blk_left_border, bool blk_right_border, bool blk_top_border, bool blk_bottom_border, int base, int height)
{

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (true)
  {
    uint32_t offset = abs(K.y - P.y) / 2;
    int2 Q = {K.x + (int)offset, K.y - (int)offset};

    if(DEBUG) printf("Block %d thread %d testing Q(%d,%d)\n", blockIdx.x, threadIdx.x, Q.x, Q.y);

    // bool Q_bottom_border = (blk_bottom_border && Q.y == blockDim.x - 1);
    bool Q_bottom_border = (blk_bottom_border && Q.y == height);
    bool Q_left_border = (blk_left_border && Q.x == 0);
    // bool Q_right_border = (blk_right_border && Q.x == blockDim.x - 1);
    bool Q_right_border = (blk_right_border && Q.x == base);
    bool Q_top_border = (blk_top_border && Q.y == 0);

    if (Q_bottom_border || Q_left_border || A_local[Q.y] >= B_local[Q.x - 1])
    {
      if (Q_right_border || Q_top_border || A_local[Q.y - 1] <= B_local[Q.x])
      {
        if (!Q_bottom_border && (Q_right_border || A_local[Q.y] <= B_local[Q.x]))
        {
          M_global[tid] = A_local[Q.y];
        }
        else
        {
          M_global[tid] = B_local[Q.x];
        }
        if(DEBUG) printf("Block %d thread %d found Q(%d,%d)\n", blockIdx.x, threadIdx.x, Q.x, Q.y);
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
__global__ void merge_k_triangles(const T *A_ptr,
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

