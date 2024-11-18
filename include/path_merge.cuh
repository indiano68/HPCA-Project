#pragma once
#include <vector>
#include <diag_search.cuh>

typedef int2 coordinate;
int constexpr THREADS_PER_BLOCK = 1024;


template <class T>
_global_ void mergeSmall_k_gpu_seq(const T *A_ptr,
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
_global_ void mergeSmall_k(const T *v_1_ptr,
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
_global_ void mergeSmall_k1(const T *v_1_ptr,
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
_global_ void mergeSmall_k2(const T *v_1_ptr,
                              const size_t v_1_size,
                              const T *v_2_ptr,
                              const size_t v_2_size,
                              T *v_out_ptr,
                              const size_t v_out_size)
{
    long long int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    coordinate K = {0, 0};
    coordinate P = {0, 0};
    coordinate Q = {0, 0};

    _shared_ int2 Q_shift;
    _shared_ T A[THREADS_PER_BLOCK + 1];
    _shared_ T B[THREADS_PER_BLOCK + 1];
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

    if ((threadIdx.x == blockDim.x - 1 && blockIdx.x != gridDim.x - 1) || thread_idx == v_out_size)
    {
        Q_shift.x = 0;
        Q_shift.y = 0;
        while (true)
        {
            size_t offset = abs((K.y - P.y)) / 2;
            Q_shift.x = K.x + offset;
            Q_shift.y = K.y - offset;
            if (Q_shift.y >= 0 && Q_shift.x <= v_2_size && (Q_shift.y == v_1_size || Q_shift.x == 0 || v_1_ptr[Q_shift.y] > v_2_ptr[Q_shift.x - 1]))
            {
                if (Q_shift.x == v_2_size || Q_shift.y == 0 || v_1_ptr[Q_shift.y - 1] <= v_2_ptr[Q_shift.x])
                {
                    break;
                }
                else
                {
                    K.x = Q_shift.x + 1;
                    K.y = Q_shift.y - 1;
                }
            }
            else
            {
                P.x = Q_shift.x - 1;
                P.y = Q_shift.y + 1;
            }
        }
    }
    if (thread_idx == v_out_size)
    {
        if (v_1_ptr[v_1_size - 1] > v_2_ptr[v_2_size - 1])
            Q_shift.y += THREADS_PER_BLOCK - 1 - threadIdx.x;
        else
            Q_shift.x += THREADS_PER_BLOCK - 1 - threadIdx.x;
    }
    __syncthreads();

    int shift_k = Q_shift.x - (int)blockDim.x + (int)threadIdx.x + 1 >= 0 ? 0 : -1 * (Q_shift.x - (int)blockDim.x + (int)threadIdx.x + 1);
    int shift_q = Q_shift.y - (int)blockDim.x + (int)threadIdx.x + 1 >= 0 ? 0 : -1 * (Q_shift.y - (int)blockDim.x + (int)threadIdx.x + 1);
    K = {Q_shift.x - (int)blockDim.x + (int)threadIdx.x + 1 + shift_k, Q_shift.y - shift_k};
    P = {Q_shift.x - shift_q, Q_shift.y - (int)blockDim.x + (int)threadIdx.x + 1 + shift_q};
    Q = {0, 0};

    A[threadIdx.x] = v_1_ptr[min(max(0, (Q_shift.y - (int)threadIdx.x)), (int)v_1_size - 1)];
    B[threadIdx.x] = v_2_ptr[min(max(0, (Q_shift.x - (int)threadIdx.x)), (int)v_2_size - 1)];
    if (threadIdx.x == 0)
    {
        A[blockDim.x] = v_1_ptr[min(max(0, (Q_shift.y - (int)blockDim.x)), (int)v_1_size - 1)];
        B[blockDim.x] = v_2_ptr[min(max(0, (Q_shift.x - (int)blockDim.x)), (int)v_2_size - 1)];
    }
    __syncthreads();

    while (true && thread_idx < v_out_size)
    {
        size_t offset = abs((K.y - P.y)) / 2;
        Q.x = K.x + offset;
        Q.y = K.y - offset;
        if (Q.y >= 0 && Q.x <= v_2_size && (Q.y == v_1_size || Q.x == 0 || A[Q_shift.y - Q.y] > B[Q_shift.x - Q.x + 1]))
        {
            if (Q.x == v_2_size || Q.y == 0 || A[Q_shift.y - Q.y + 1] <= B[Q_shift.x - Q.x])
            {
                if (Q.y < v_1_size && (Q.x == v_2_size || A[Q_shift.y - Q.y] <= B[Q_shift.x - Q.x]))
                {
                    v_out_ptr[thread_idx] = A[Q_shift.y - Q.y];
                }
                else
                {
                    v_out_ptr[thread_idx] = B[Q_shift.x - Q.x];
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
_global_ void merge_k_gpu_triangles(const T *A_ptr,
                                            const size_t A_size,
                                            const T *B_ptr,
                                            const size_t B_size,
                                            T *M_ptr)
{

  //static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");

  _shared_ T A_shared[THREADS_PER_BLOCK + 1], B_shared[THREADS_PER_BLOCK + 1];
  _shared_ int2 Q_base;

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
_global_ void partition_k_gpu(const T *A_ptr,
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
_global_ void merge_k_gpu_squares(const T *A_ptr,
                                            const size_t A_size,
                                            const T *B_ptr,
                                            const size_t B_size,
                                            T *M_ptr,
                                            const int2 *Q_global)
{

  //assert(blockDim.x >= 4);

  _shared_ T shared_mem[THREADS_PER_BLOCK + 4];

  int2 Q_next = Q_global[blockIdx.x];
  int2 Q_prev = (blockIdx.x > 0) ? Q_global[blockIdx.x - 1] : make_int2(0, 0);

  int x_start = Q_prev.x;
  int y_start = Q_prev.y;
  int x_end = Q_next.x;
  int y_end = Q_next.y;

  int base = Q_next.x - Q_prev.x;
  int height = Q_next.y - Q_prev.y;

  if(blockIdx.x != 0)
  {
    if(height == 0)
    {
      base--;
      x_start++;
    }
    else if(base == 0)
    {
      height--;
      y_start++;
    }
  }

  T *A_local = shared_mem;
  T *B_local = shared_mem + height + 2;

  if(threadIdx.x < height)
  {
    A_local[threadIdx.x + 1] = A_ptr[y_start + threadIdx.x];
  }
  else if(threadIdx.x < height + base)
  {
    B_local[threadIdx.x - height + 1] = B_ptr[x_start + threadIdx.x - height];
  }

  if(threadIdx.x == 0)
  {
    A_local[0] = (y_start > 0) ? A_ptr[y_start - 1] : 0;
  }
  else if(threadIdx.x == 1)
  {
    A_local[height + 1] = (y_end < A_size) ? A_ptr[y_end] : 0;
  }
  else if(threadIdx.x == 2)
  {
    B_local[0] = (x_start > 0) ? B_ptr[x_start - 1] : 0;
  }
  else if(threadIdx.x == 3)
  {
    B_local[base + 1] = (x_end < B_size) ? B_ptr[x_end] : 0;
  }

  // if(threadIdx.x < 2)
  // {
  //   int idx = threadIdx.x * (height + 1);
  //   A_local[idx] = (y_start > 0 && threadIdx.x == 0) ? A_ptr[y_start - 1] : (y_end < A_size && threadIdx.x == 1) ? A_ptr[y_end] : 0;
  // }

  __syncthreads();

  if(DEBUG) print_shared(A_local, B_local, base, height);

  int2 K, P;

  bool blk_left_border = (x_start == 0);
  bool blk_top_border = (y_start == 0);
  bool blk_bottom_border = (y_end == A_size);
  bool blk_right_border = (x_end == B_size);

  int reverse_tid = blockDim.x - threadIdx.x - 1;

  K.x = base >= reverse_tid ? base - reverse_tid : 0;
  K.y = base >= reverse_tid ? height : height + base - reverse_tid;

  P.x = height >= reverse_tid ? base : base + height - reverse_tid;
  P.y = height >= reverse_tid ? height - reverse_tid : 0;

  block_bin_search(A_local + 1, B_local + 1, K, P, M_ptr, blk_left_border, blk_right_border, blk_top_border, blk_bottom_border, base, height);

  return;
}


template <class T>
_global_ void merge_k_gpu_squares_v2(const T *A_ptr,
                                            const size_t A_size,
                                            const T *B_ptr,
                                            const size_t B_size,
                                            T *M_ptr)
{

  //assert(blockDim.x >= 4);

  _shared_ T shared_mem[THREADS_PER_BLOCK + 4];
  _shared_ int2 Q_prev = {0, 0}, Q_next = {0, 0};

  if (threadIdx.x == 0)
  {
    int diag_prev = (blockIdx.x != 0) ? THREADS_PER_BLOCK * blockIdx.x - 1 : 0;
    int2 K_explorative = {0, 0}, P_explorative = {0, 0};

    K_explorative.x = (diag_prev > A_size) ? diag_prev - A_size : 0;
    K_explorative.y = (diag_prev < A_size) ? diag_prev : A_size;
    P_explorative.x = (diag_prev < B_size) ? diag_prev : B_size;
    P_explorative.y = (diag_prev > B_size) ? diag_prev - B_size : 0;

    Q_prev = explorative_search(A_ptr, A_size, B_ptr, B_size, K_explorative, P_explorative);
  }
  else if (threadIdx.x == blockDim.x - 1)
  {
    int diag_next = THREADS_PER_BLOCK * (blockIdx.x + 1) - 1;
    int2 K_explorative = {0, 0}, P_explorative = {0, 0};

    K_explorative.x = (diag_next > A_size) ? diag_next - A_size : 0;
    K_explorative.y = (diag_next < A_size) ? diag_next : A_size;
    P_explorative.x = (diag_next < B_size) ? diag_next : B_size;
    P_explorative.y = (diag_next > B_size) ? diag_next - B_size : 0;

    Q_next = explorative_search(A_ptr, A_size, B_ptr, B_size, K_explorative, P_explorative);
  }

  __syncthreads();

  //int2 Q_next = Q_global[blockIdx.x];
  //int2 Q_prev = (blockIdx.x > 0) ? Q_global[blockIdx.x - 1] : make_int2(0, 0);

  int x_start = Q_prev.x;
  int y_start = Q_prev.y;
  int x_end = Q_next.x;
  int y_end = Q_next.y;

  int base = Q_next.x - Q_prev.x;
  int height = Q_next.y - Q_prev.y;

  if(blockIdx.x != 0)
  {
    if(height == 0)
    {
      base--;
      x_start++;
    }
    else if(base == 0)
    {
      height--;
      y_start++;
    }
  }

  T *A_local = shared_mem;
  T *B_local = shared_mem + height + 2;

  if(threadIdx.x < height)
  {
    A_local[threadIdx.x + 1] = A_ptr[y_start + threadIdx.x];
  }
  else if(threadIdx.x < height + base)
  {
    B_local[threadIdx.x - height + 1] = B_ptr[x_start + threadIdx.x - height];
  }

  if(threadIdx.x == 0)
  {
    A_local[0] = (y_start > 0) ? A_ptr[y_start - 1] : 0;
  }
  else if(threadIdx.x == 1)
  {
    A_local[height + 1] = (y_end < A_size) ? A_ptr[y_end] : 0;
  }
  else if(threadIdx.x == 2)
  {
    B_local[0] = (x_start > 0) ? B_ptr[x_start - 1] : 0;
  }
  else if(threadIdx.x == 3)
  {
    B_local[base + 1] = (x_end < B_size) ? B_ptr[x_end] : 0;
  }

  // if(threadIdx.x < 2)
  // {
  //   int idx = threadIdx.x * (height + 1);
  //   A_local[idx] = (y_start > 0 && threadIdx.x == 0) ? A_ptr[y_start - 1] : (y_end < A_size && threadIdx.x == 1) ? A_ptr[y_end] : 0;
  // }

  __syncthreads();

  if(DEBUG) print_shared(A_local, B_local, base, height);

  int2 K, P;

  bool blk_left_border = (x_start == 0);
  bool blk_top_border = (y_start == 0);
  bool blk_bottom_border = (y_end == A_size);
  bool blk_right_border = (x_end == B_size);

  int reverse_tid = blockDim.x - threadIdx.x - 1;

  K.x = base >= reverse_tid ? base - reverse_tid : 0;
  K.y = base >= reverse_tid ? height : height + base - reverse_tid;

  P.x = height >= reverse_tid ? base : base + height - reverse_tid;
  P.y = height >= reverse_tid ? height - reverse_tid : 0;

  block_bin_search(A_local + 1, B_local + 1, K, P, M_ptr, blk_left_border, blk_right_border, blk_top_border, blk_bottom_border, base, height);

  return;
}