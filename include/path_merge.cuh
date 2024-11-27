#pragma once
#include <vector>
#include <diag_search.cuh>
#include <partition.cuh>

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

  if constexpr(DEBUG) if(threadIdx.x == 0)
  {
    printf("Block %d, Q_next(%d,%d) Q_prev(%d,%d), base = %d, height = %d\n", blockIdx.x, Q_next.x, Q_next.y, Q_prev.x, Q_prev.y, base, height);
  }

  //shared memory loading
  T *A_block = shared_mem_block;
  T *B_block = shared_mem_block + height;
  {
    int tiles;
    for(tiles = 0; tiles < height / TILE_SIZE; tiles++)
    {
      A_block[threadIdx.x + tiles * TILE_SIZE] = A_ptr[y_start + threadIdx.x + tiles * TILE_SIZE];
    }

    if (threadIdx.x + tiles * TILE_SIZE < height)
    {
      A_block[threadIdx.x + tiles * TILE_SIZE] = A_ptr[y_start + threadIdx.x + tiles * TILE_SIZE];
    }

    for(tiles = 0; tiles < base / TILE_SIZE; tiles++)
    {
      B_block[threadIdx.x + tiles * TILE_SIZE] = B_ptr[x_start + threadIdx.x + tiles * TILE_SIZE];
    }

    if (threadIdx.x + tiles * TILE_SIZE < base)
    {
      B_block[threadIdx.x + tiles * TILE_SIZE] = B_ptr[x_start + threadIdx.x + tiles * TILE_SIZE];
    }
    if constexpr(DEBUG) print_shared(A_block, B_block, base, height);
    __syncthreads();
  }

  //keeps track of bottom-right corner of the current tile (in tile coordinates)
  __shared__ int2 Q_tile_t;

  T *A_tile = A_block;
  T *B_tile = B_block;

  int leftover_base = base, leftover_height = height;

  //loop over tiles
  for(int tile = 0; tile < TILES_PER_BLOCK; tile++)
  {
    int tile_base, tile_height;
    bool tile_bottom_border, tile_right_border; 

    int block_diag_idx = tile * TILE_SIZE + threadIdx.x;

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
    K.y = threadIdx.x <= tile_height ? threadIdx.x : tile_height;

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

    if constexpr(DEBUG) if(threadIdx.x == 0 || threadIdx.x == last_tid_of_block)
    {
        printf("Block %d, tile %d, block_diag_idx = %d, search range: K(%d,%d) - P(%d,%d), found Q_tile(%d,%d), leftover_base = %d, leftover_height = %d, A_tile[0] = %d, B_tile[0] = %d\n", blockIdx.x, tile, block_diag_idx, K.x, K.y, P.x, P.y, Q_tile_t.x, Q_tile_t.y, leftover_base, leftover_height, A_tile[0], B_tile[0]);
    }

    leftover_base -= Q_tile_t.x;
    leftover_height -= Q_tile_t.y;

    A_tile += Q_tile_t.y;
    B_tile += Q_tile_t.x;
  }

  return;  
}

template <class T>
__global__ void merge_k_gpu_serial_tile(const T *A_ptr,
                                        const size_t A_size,
                                        const T *B_ptr,
                                        const size_t B_size,
                                        T *M_ptr,
                                        const int2 *Q_global)
{

  __shared__ T shared_mem_block[BOX_SIZE];

  //int last_tid_of_block = TILE_SIZE - 1;

  int2 Q_next = Q_global[blockIdx.x];
  int2 Q_prev = (blockIdx.x > 0) ? Q_global[blockIdx.x - 1] : make_int2(0, 0);

  int x_start = Q_prev.x;
  int y_start = Q_prev.y;

  int box_base = Q_next.x - Q_prev.x;
  int box_height = Q_next.y - Q_prev.y;

  if constexpr(DEBUG) if(threadIdx.x == 0)
  {
    printf("Block %d, Q_next(%d,%d) Q_prev(%d,%d), box_base = %d, box_height = %d\n", blockIdx.x, Q_next.x, Q_next.y, Q_prev.x, Q_prev.y, box_base, box_height);
  }

  //shared memory loading
  T *A_block = shared_mem_block;
  T *B_block = shared_mem_block + box_height;
  {
    int tiles;
    for(tiles = 0; tiles < box_height / THREADS_PER_BOX; tiles++)
    {
      A_block[threadIdx.x + tiles * THREADS_PER_BOX] = A_ptr[y_start + threadIdx.x + tiles * THREADS_PER_BOX];
    }

    if (threadIdx.x + tiles * THREADS_PER_BOX < box_height)
    {
      A_block[threadIdx.x + tiles * THREADS_PER_BOX] = A_ptr[y_start + threadIdx.x + tiles * THREADS_PER_BOX];
    }

    for(tiles = 0; tiles < box_base / THREADS_PER_BOX; tiles++)
    {
      B_block[threadIdx.x + tiles * THREADS_PER_BOX] = B_ptr[x_start + threadIdx.x + tiles * THREADS_PER_BOX];
    }

    if (threadIdx.x + tiles * THREADS_PER_BOX < box_base)
    {
      B_block[threadIdx.x + tiles * THREADS_PER_BOX] = B_ptr[x_start + threadIdx.x + tiles * THREADS_PER_BOX];
    }
    if constexpr(DEBUG) print_shared(A_block, B_block, box_base, box_height);
    __syncthreads();
  }

  int2 Q_start = partition_box(A_block, box_height, B_block, box_base, WORK_PER_THREAD);

  if constexpr(DEBUG) printf("MERGE_SERIAL_TILE: Block %d thread %d, Q_start(%d,%d)\n", blockIdx.x, threadIdx.x, Q_start.x, Q_start.y);

  bool thread_within_bounds = threadIdx.x * WORK_PER_THREAD < box_base + box_height;

  if(thread_within_bounds)
  {
    unsigned thread_work = min(WORK_PER_THREAD, box_base + box_height - threadIdx.x * WORK_PER_THREAD);

    if constexpr(DEBUG) printf("Block %d thread %d in bounds, thread_work = %d\n", blockIdx.x, threadIdx.x, thread_work);

    T* A_tile = A_block + Q_start.y;
    T* B_tile = B_block + Q_start.x;

    unsigned i = 0, j = 0;
    //perform serial tile merge
    for(int item = 0; item < thread_work; item++)
    {
      bool insert_A = (Q_start.y + i < box_height) && (Q_start.x + j >= box_base || A_tile[i] < B_tile[j]);

      if constexpr(DEBUG) printf("Block %d thread %d, comparing A_tile[%d] < B_tile[%d], A_tile[%d] = %d, B_tile[%d] = %d\n", blockIdx.x, threadIdx.x, i, j, i, A_tile[i], j, B_tile[j]);
      if(insert_A)
      {
        M_ptr[Q_prev.x + Q_prev.y + threadIdx.x * WORK_PER_THREAD + item] = A_tile[i];
        i++;
      }
      else
      {
        M_ptr[Q_prev.x + Q_prev.y + threadIdx.x * WORK_PER_THREAD + item] = B_tile[j];
        j++;
      }
    }
  }

  return;  
}

template <class T>
__global__ void merge_k_gpu_serial_tile_shared(const T *A_ptr,
                                        const size_t A_size,
                                        const T *B_ptr,
                                        const size_t B_size,
                                        T *M_ptr,
                                        const int2 *Q_global)
{

  //TODO: use less shared memory
  __shared__ T shared_mem_block[BOX_SIZE];
  __shared__ T M_shared[BOX_SIZE];

  int2 Q_next = Q_global[blockIdx.x];
  int2 Q_prev = (blockIdx.x > 0) ? Q_global[blockIdx.x - 1] : make_int2(0, 0);

  unsigned box_x_start = Q_prev.x;
  unsigned box_y_start = Q_prev.y;

  unsigned box_base = Q_next.x - Q_prev.x;
  unsigned box_height = Q_next.y - Q_prev.y;

  if constexpr(DEBUG) if(threadIdx.x == 0)
  {
    printf("Block %d, Q_next(%d,%d) Q_prev(%d,%d), box_base = %d, box_height = %d\n", blockIdx.x, Q_next.x, Q_next.y, Q_prev.x, Q_prev.y, box_base, box_height);
  }

  //shared memory loading
  T *A_block = shared_mem_block;
  T *B_block = shared_mem_block + box_height;
  
  // Load A_block
  for (unsigned i = threadIdx.x; i < box_height; i += THREADS_PER_BOX) {
      A_block[i] = A_ptr[box_y_start + i];
  }

  // Load B_block
  for (unsigned i = threadIdx.x; i < box_base; i += THREADS_PER_BOX) {
      B_block[i] = B_ptr[box_x_start + i];
  }

  if constexpr(DEBUG) print_shared(A_block, B_block, box_base, box_height);
  __syncthreads();

  //fine grained partitioning in shared memory
  int2 Q_start = partition_box(A_block, box_height, B_block, box_base, WORK_PER_THREAD);

  //here we exclude threads that do not have even a single element to merge (i.e. threads whose tile is completely out of bounds)
  unsigned effective_box_size = box_base + box_height; //could be less than BOX_SIZE
  bool thread_within_bounds = threadIdx.x * WORK_PER_THREAD < effective_box_size;

  if(thread_within_bounds)
  {

    T* A_tile = A_block + Q_start.y;
    T* B_tile = B_block + Q_start.x;

    //perform serial tile merge
    unsigned i = 0, j = 0;
#pragma unroll
    //we use WORK_PER_THREAD here because shared memory is allocated even for threads that are not in bounds
    //for(int item = 0; item < thread_work; item++)
    for(unsigned item = 0; item < WORK_PER_THREAD; item++)
    {
      bool insert_A = (Q_start.y + i < box_height) && (Q_start.x + j >= box_base || A_tile[i] < B_tile[j]);

      if(insert_A)
      {
        M_shared[threadIdx.x * WORK_PER_THREAD + item] = A_tile[i];
        i++;
      }
      else
      {
        M_shared[threadIdx.x * WORK_PER_THREAD + item] = B_tile[j];
        j++;
      }
    }
    //not needed here (i believe)
    // __syncthreads();

    //when storing to global memory we store only the elements that are in bounds (some threads may have less than WORK_PER_THREAD elements to merge)
    unsigned thread_work = min(WORK_PER_THREAD, effective_box_size - threadIdx.x * WORK_PER_THREAD);
    for(unsigned item = 0; item < thread_work; item++)
    {
      M_ptr[Q_prev.x + Q_prev.y + threadIdx.x * WORK_PER_THREAD + item] = M_shared[threadIdx.x * WORK_PER_THREAD + item];
    }

  }

  return;  
}