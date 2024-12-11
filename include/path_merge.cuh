#pragma once

#include <diag_search.cuh>
#include <partition.cuh>

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
__global__ void mergeLarge_naive_k(const T *v_1_ptr,
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
__device__ __forceinline__ int2 bin_search_window(const T *A_local, const T *B_local, 
                                                  int2 K, int2 P, T *M_global, 
                                                  int tile_base, int tile_height, 
                                                  bool tile_bottom_border, bool tile_right_border,
                                                  int block_diag_idx)
{

  int M_idx = block_diag_idx + blockIdx.x * TILE_SIZE * TILES_PER_BLOCK;

  while (true)
  {
    uint32_t offset = (K.y - P.y) / 2;
    int2 Q = {K.x + (int)offset, K.y - (int)offset};

    bool Q_bottom_border = (Q.y == tile_height) && tile_bottom_border;
    bool Q_right_border = (Q.x == tile_base) && tile_right_border;
    bool Q_left_border = (Q.x == 0);
    bool Q_top_border = (Q.y == 0);

    if (Q_bottom_border || Q_left_border || A_local[Q.y] > B_local[Q.x - 1])
    {
      if (Q_right_border || Q_top_border || A_local[Q.y - 1] <= B_local[Q.x])
      {
        if (!Q_bottom_border && (Q_right_border || A_local[Q.y] <= B_local[Q.x]))
        {
          M_global[M_idx] = A_local[Q.y];
          //if inserting from A, merge path goes down
          Q.y++;
        }
        else
        {
          M_global[M_idx] = B_local[Q.x];
          //if inserting from B, merge path goes right
          Q.x++;
        }
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
__global__ void mergeLarge_window_k(const T *A_ptr,
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

  //shared memory loading
  T *A_block = shared_mem_block;
  T *B_block = shared_mem_block + height;

  for (unsigned i = threadIdx.x; i < height; i += TILE_SIZE) {
      A_block[i] = A_ptr[y_start + i];
  }

  // Load B_block
  for (unsigned i = threadIdx.x; i < base; i += TILE_SIZE) {
      B_block[i] = B_ptr[x_start + i];
  }
  __syncthreads();

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
      Q_temp = bin_search_window(A_tile, B_tile, K, P, M_ptr, tile_base, tile_height, tile_bottom_border, tile_right_border, block_diag_idx);
    }

    if(threadIdx.x == last_tid_of_block)
    {
      Q_tile_t = Q_temp;
    }
    __syncthreads();

    leftover_base -= Q_tile_t.x;
    leftover_height -= Q_tile_t.y;

    A_tile += Q_tile_t.y;
    B_tile += Q_tile_t.x;
  }

  return;  
}

template <class T>
__global__ void mergeLarge_tiled_k(const T *A_ptr,
                                   const size_t A_size,
                                   const T *B_ptr,
                                   const size_t B_size,
                                   T *M_ptr,
                                   const int2 *Q_global)
{

  __shared__ T shared_mem_block[BOX_SIZE];

  int2 Q_next = Q_global[blockIdx.x];
  int2 Q_prev = (blockIdx.x > 0) ? Q_global[blockIdx.x - 1] : make_int2(0, 0);

  unsigned box_x_start = Q_prev.x;
  unsigned box_y_start = Q_prev.y;

  unsigned box_base = Q_next.x - Q_prev.x;
  unsigned box_height = Q_next.y - Q_prev.y;

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

  __syncthreads();

  //fine grained partitioning in shared memory
  int2 Q_start = partition_box(A_block, box_height, B_block, box_base, WORK_PER_THREAD);

  T* A_tile = A_block + Q_start.y;
  T* B_tile = B_block + Q_start.x;

  T M_tile[WORK_PER_THREAD];

  //perform serial tile merge
  unsigned i = 0, j = 0;
#pragma unroll
  //we don't check bounds, end of shared memory will contain garbage for threads that are not in bounds
  for(unsigned item = 0; item < WORK_PER_THREAD; item++)
  {
    bool insert_A = (Q_start.y + i < box_height) && (Q_start.x + j >= box_base || A_tile[i] < B_tile[j]);

    if(insert_A)
    {
      M_tile[item] = A_tile[i];
      i++;
    }
    else
    {
      M_tile[item] = B_tile[j];
      j++;
    }
  }
  __syncthreads();

  //copy to shared memory (garbage for threads that are not in bounds)
  for(unsigned item = 0; item < WORK_PER_THREAD; item++)
  {
    shared_mem_block[threadIdx.x * WORK_PER_THREAD + item] = M_tile[item];
  }
  __syncthreads();

  //store to global memory, here we check bounds
  unsigned effective_box_size = box_base + box_height; //could be less than BOX_SIZE
  unsigned M_idx_start = Q_prev.x + Q_prev.y;
  for (unsigned item = threadIdx.x; item < effective_box_size; item += THREADS_PER_BOX)
  {
    M_ptr[M_idx_start + item] = shared_mem_block[item];
  }

  return;  
}