#pragma once
#include <vector>
#include <diag_search.cuh>

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
__global__ void merge_k_blocked(const T *v_1_ptr,
                              const int v_1_size,
                              const T *v_2_ptr,
                              const int v_2_size,
                              T *v_out_ptr,
                              const int v_out_size,
                              const int2* v_Q_ptr)
{
    int2 K = {0, 0};
    int2 P = {0, 0};
    int2 Q = {0, 0};
    int2 Q_base = v_Q_ptr[blockIdx.x];
    int2 Q_end;
    if(blockIdx.x != gridDim.x-1)
    {
        Q_end = v_Q_ptr[blockIdx.x +1];
    }
    else
    {
        Q_end.x = v_2_size;
        Q_end.y = v_1_size;
    }
    int v_1_blocked_size = Q_end.y - Q_base.y;
    int v_2_blocked_size = Q_end.x - Q_base.x;
    if (threadIdx.x > v_1_blocked_size)
    {
        K.x = threadIdx.x - v_1_blocked_size;
        K.y = v_1_blocked_size;
        P.x = v_1_blocked_size;
        P.y = threadIdx.x - v_1_blocked_size;
    }
    else
    {
        K.y = threadIdx.x;
        P.x = threadIdx.x;
    }
    __shared__ T total [THREADS_PER_BLOCK];

    T * v_blocked_1_ptr = total;
    T * v_blocked_2_ptr = total+v_1_blocked_size;

    if(threadIdx.x<v_1_blocked_size)
        v_blocked_1_ptr[threadIdx.x] = v_1_ptr[Q_base.y + threadIdx.x];
    else if(threadIdx.x<v_1_blocked_size+v_2_blocked_size)
        v_blocked_2_ptr[threadIdx.x - v_1_blocked_size] = v_2_ptr[Q_base.x + threadIdx.x-v_1_blocked_size];
    __syncthreads();

    while (true && threadIdx.x < v_1_blocked_size+v_2_blocked_size)
    {
        size_t offset = abs((K.y - P.y)) / 2;
        Q.x = K.x + offset;
        Q.y = K.y - offset;
        if (Q.y >= 0 && Q.x <= v_2_blocked_size && (Q.y == v_1_blocked_size || Q.x == 0 || v_blocked_1_ptr[Q.y] > v_blocked_2_ptr[Q.x - 1]))
        {
            if (Q.x == v_2_blocked_size || Q.y == 0 || v_blocked_1_ptr[Q.y - 1] <= v_blocked_2_ptr[Q.x])
            {
                if (Q.y < v_1_blocked_size && (Q.x == v_2_blocked_size || v_blocked_1_ptr[Q.y] <= v_blocked_2_ptr[Q.x]))
                {
                    v_out_ptr[Q_base.x + Q_base.y + threadIdx.x] = v_blocked_1_ptr[Q.y];
                }
                else
                {
                    v_out_ptr[Q_base.x + Q_base.y + threadIdx.x] = v_blocked_2_ptr[Q.x];
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
__global__ void merge_k_triangles(const T *A_ptr,
                                  const size_t A_size,
                                  const T *B_ptr,
                                  const size_t B_size,
                                  T *M_ptr)
{

    // static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");

    __shared__ T A_shared[THREADS_PER_BLOCK + 1], B_shared[THREADS_PER_BLOCK + 1];
    __shared__ int2 Q_base;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int last_tid_of_block = blockDim.x * (blockIdx.x + 1) - 1;

    if (tid == last_tid_of_block)
    {
        int2 K_explorative = {0, 0}, P_explorative = {0, 0};

        K_explorative.x = (tid > A_size) ? tid - A_size : 0;
        K_explorative.y = (tid < A_size) ? tid : A_size;
        P_explorative.x = (tid < B_size) ? tid : B_size;
        P_explorative.y = (tid > B_size) ? tid - B_size : 0;

        Q_base = explorative_search(A_ptr, A_size, B_ptr, B_size, K_explorative, P_explorative);

        if constexpr(DEBUG)
            printf("Explorative search (block %d) found Q_base(%d,%d)\n", blockIdx.x, Q_base.x, Q_base.y);
    }
    __syncthreads();

    int x_end = Q_base.x;
    int y_end = Q_base.y;

    int x_start = x_end >= blockDim.x ? x_end - blockDim.x + 1 : 0;
    int y_start = y_end >= blockDim.x ? y_end - blockDim.x + 1 : 0;

    bool blk_left_border = (x_start == 0);
    bool blk_top_border = (y_start == 0);
    bool blk_bottom_border = (y_end == A_size);
    bool blk_right_border = (x_end == B_size);

    A_shared[threadIdx.x + 1] = y_start + threadIdx.x < A_size ? A_ptr[y_start + threadIdx.x] : 0;
    B_shared[threadIdx.x + 1] = x_start + threadIdx.x < B_size ? B_ptr[x_start + threadIdx.x] : 0;

    if (threadIdx.x == 0)
    {
        A_shared[0] = (y_start > 0) ? A_ptr[y_start - 1] : 0;
        B_shared[0] = (x_start > 0) ? B_ptr[x_start - 1] : 0;
    }

    int base = x_end - x_start;   // base del triangolo
    int height = y_end - y_start; // altezza del triangolo

    if (DEBUG && threadIdx.x == 0)
        printf("Block %d, base = %d, height = %d\n", blockIdx.x, base, height);

    // complementare di threadIdx.x nel blocco
    int reverse_tid = blockDim.x - threadIdx.x - 1;

    int2 K, P;

    K.x = base >= reverse_tid ? base - reverse_tid : 0;
    K.y = base >= reverse_tid ? height : height + base - reverse_tid;

    P.x = height >= reverse_tid ? base : base + height - reverse_tid;
    P.y = height >= reverse_tid ? height - reverse_tid : 0;

    if (DEBUG)
        printf("Block %d thread %d search range: K(%d,%d) P(%d,%d)\n", blockIdx.x, threadIdx.x, K.x, K.y, P.x, P.y);

    __syncthreads();

    block_bin_search_triangles(A_shared + 1, B_shared + 1, K, P, M_ptr, blk_left_border, blk_right_border, blk_top_border, blk_bottom_border, base, height);

    if (DEBUG)
        printf("Block %d thread %d finished binsearch\n", blockIdx.x, threadIdx.x);

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

  __shared__ T shared_mem[THREADS_PER_BLOCK];
  
  int2 Q_next = Q_global[blockIdx.x];
  int2 Q_prev = (blockIdx.x > 0) ? Q_global[blockIdx.x - 1] : make_int2(0, 0);

  int x_start = Q_prev.x;
  int y_start = Q_prev.y;

  int base = Q_next.x - Q_prev.x;
  int height = Q_next.y - Q_prev.y;

  T *A_local = shared_mem;
  T *B_local = shared_mem + height;

  if(threadIdx.x < height)
  {
    A_local[threadIdx.x] = A_ptr[y_start + threadIdx.x];
  }
  else if(threadIdx.x < height + base)
  {
    B_local[threadIdx.x - height] = B_ptr[x_start + threadIdx.x - height];
  }

  //__syncthreads();


  if constexpr(DEBUG) print_shared(A_local, B_local, base, height);

  int2 K, P;

  K.x = threadIdx.x <= height ? 0 : threadIdx.x - height;
  K.y = threadIdx.x <= height ? threadIdx.x : height;

  P.x = threadIdx.x <= base ? threadIdx.x : base;
  P.y = threadIdx.x <= base ? 0 : threadIdx.x - base;

  if constexpr(DEBUG)
  {
    if(threadIdx.x == 0)
    {
      printf("Block %d, Q_next(%d,%d) Q_prev(%d,%d), base = %d, height = %d\n", blockIdx.x, Q_next.x, Q_next.y, Q_prev.x, Q_prev.y, base, height);
      printf("Block %d thread %d search range: K(%d,%d) P(%d,%d)\n", blockIdx.x, threadIdx.x, K.x, K.y, P.x, P.y);
    }
  } 

  __syncthreads();

  // __shared__ T M_shared[THREADS_PER_BLOCK];

  // if(threadIdx.x < base + height)
  // {
  //   block_bin_search_shared(A_local, B_local, K, P, M_ptr, base, height, M_shared);

  //   int M_idx = threadIdx.x + blockIdx.x * blockDim.x - 1 * (blockIdx.x != 0);

  //   M_ptr[M_idx] = M_shared[threadIdx.x];
  // }

  if(threadIdx.x < base + height)
  {
    block_bin_search(A_local, B_local, K, P, M_ptr, base, height);
  }

  return;
    
}

template <class T>
__global__ void merge_k_gpu_window(const T *A_ptr,
                                    const size_t A_size,
                                    const T *B_ptr,
                                    const size_t B_size,
                                    T *M_ptr,
                                    const int2 *Q_global)
{

  __shared__ T shared_mem_block[TILE_SIZE * TILES_PER_BLOCK];

  int last_tid_of_block = TILE_SIZE - 1;

  int2 Q_next = Q_global[blockIdx.x];
  int2 Q_prev = (blockIdx.x > 0) ? Q_global[blockIdx.x - 1] : make_int2(0, 0);

  int x_start = Q_prev.x;
  int y_start = Q_prev.y;

  int base = Q_next.x - Q_prev.x;
  int height = Q_next.y - Q_prev.y;

  if constexpr(DEBUG)
  {
    if(threadIdx.x == 0)
    {
      printf("Block %d, Q_next(%d,%d) Q_prev(%d,%d), base = %d, height = %d\n", blockIdx.x, Q_next.x, Q_next.y, Q_prev.x, Q_prev.y, base, height);
    }
  }  

  T *A_block = shared_mem_block;
  T *B_block = shared_mem_block + height;

  __shared__ int2 Q_tile_t;

  int i;
  for(i = 0; i < height / TILE_SIZE; i++)
  {
    A_block[threadIdx.x + i * TILE_SIZE] = A_ptr[y_start + threadIdx.x + i * TILE_SIZE];
  }

  if (threadIdx.x + i * TILE_SIZE < height)
  {
    A_block[threadIdx.x + i * TILE_SIZE] = A_ptr[y_start + threadIdx.x + i * TILE_SIZE];
  }

  for(i = 0; i < base / TILE_SIZE; i++)
  {
    B_block[threadIdx.x + i * TILE_SIZE] = B_ptr[x_start + threadIdx.x + i * TILE_SIZE];
  }

  if (threadIdx.x + i * TILE_SIZE < base)
  {
    B_block[threadIdx.x + i * TILE_SIZE] = B_ptr[x_start + threadIdx.x + i * TILE_SIZE];
  }

  if constexpr(DEBUG) print_shared(A_block, B_block, base, height);

  int leftover_base = base, leftover_height = height;

  T *A_tile = A_block;
  T *B_tile = B_block;

  bool tile_bottom_border, tile_right_border; 

  __syncthreads();
  for(int tile = 0; tile < TILES_PER_BLOCK; tile++)
  {
    int block_diag_idx = tile * TILE_SIZE + threadIdx.x;

    int tile_base, tile_height;

    if(leftover_base <= TILE_SIZE)
    {
      tile_base = leftover_base;
      tile_right_border = true;
    }
    else
    {
      tile_base = TILE_SIZE;
      tile_right_border = false;
    }

    if(leftover_height <= TILE_SIZE)
    {
      tile_height = leftover_height;
      tile_bottom_border = true;
    }
    else
    {
      tile_height = TILE_SIZE;
      tile_bottom_border = false;
    }

    int2 K, P;

    K.x = threadIdx.x <= tile_height ? 0 : threadIdx.x - tile_height;
    K.y = threadIdx.x <= tile_height ? threadIdx.x :tile_height;

    P.x = threadIdx.x <= tile_base ? threadIdx.x : tile_base;
    P.y = threadIdx.x <= tile_base ? 0 : threadIdx.x - tile_base;

    int2 Q_temp;
    if(block_diag_idx < height + base)
    {
      Q_temp = block_bin_search_tiled(A_tile, B_tile, K, P, M_ptr, tile_base, tile_height, tile_bottom_border, tile_right_border, block_diag_idx);
    }

    if(threadIdx.x == last_tid_of_block)
    {
      Q_tile_t = Q_temp;
    }
    __syncthreads();

    if constexpr(DEBUG)
    {
      if(threadIdx.x == 0 || threadIdx.x == last_tid_of_block)
      {
        printf("Block %d, tile %d, block_diag_idx = %d, search range: K(%d,%d) - P(%d,%d), found Q_tile(%d,%d), leftover_base = %d, leftover_height = %d, A_tile[0] = %d, B_tile[0] = %d\n", blockIdx.x, tile, block_diag_idx, K.x, K.y, P.x, P.y, Q_tile_t.x, Q_tile_t.y, leftover_base, leftover_height, A_tile[0], B_tile[0]);
      }
    }

    leftover_base -= Q_tile_t.x;
    leftover_height -= Q_tile_t.y;

    A_tile += Q_tile_t.y;
    B_tile += Q_tile_t.x;
  }

  return;  
}
