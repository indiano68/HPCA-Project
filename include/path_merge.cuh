#pragma once

#include <partition.cuh>

//gpuLargeMerge_window_k
constexpr unsigned BLK_SIZE_WINDOW_K = 512;
constexpr unsigned WINDOWS_PER_BLK = 12;

//gpuMergeLarge_tiled_k
constexpr unsigned BLK_SIZE_TILED_K = 512; // CUDA block size
constexpr unsigned WORK_PER_THREAD = 15; // number of consecutive elements to process per thread
constexpr unsigned WORK_PER_BLK = BLK_SIZE_TILED_K * WORK_PER_THREAD; // number of elements to process per block

//gpuMergeLarge_naive_k
constexpr unsigned BLK_SIZE_NAIVE_K = 512;

template <class T>
__global__ void mergeSmall_k(const T *v_1_ptr,
                             const size_t v_1_size,
                             const T *v_2_ptr,
                             const size_t v_2_size,
                             T *v_out_ptr,
                             const size_t v_out_size)
{
    int thread_idx = threadIdx.x;
    int2 K = {0, 0};
    int2 P = {0, 0};
    int2 Q = {0, 0};
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

/*
 * Naive extension of MergeSmall to large arrays by replacing threadIdx.x with blockIdx.x * blockDim.x + threadIdx.x
*/
template <class T>
__global__ void mergeLarge_naive_k(const T *v_1_ptr,
                                   const size_t v_1_size,
                                   const T *v_2_ptr,
                                   const size_t v_2_size,
                                   T *v_out_ptr)
{
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned v_out_size = v_1_size + v_2_size;

    int2 K = {0, 0};
    int2 P = {0, 0};
    int2 Q = {0, 0};
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

/**
 * @brief Device function to perform binary search on a single window in shared memory, used in mergeLarge_window_k.
 * As soon as it finds the element to insert, it copies it directly from shared memory to global memory.
 * It returns the coordinates of the next point along the merge path. This information is only actually used by the last thread of the block,
 * which communicates it to the other threads in the block so that the next window can be processed.
 * The bound checks have been replaced by boolean checks to let the threads know if the point they are processing is on the border of the window.
 */
template <class T>
__device__ __forceinline__ int2 bin_search_window(const T *A_local, const T *B_local, 
                                                  int2 K, int2 P, T *M_global, 
                                                  int window_base, int window_heigth, 
                                                  bool window_bottom_border, bool window_right_border,
                                                  int block_diag_idx)
{
  //block_diag_idx is the index of the current diagonal in the block, M_idx is thw global index where the current thread should write
  int M_idx = block_diag_idx + blockIdx.x * BLK_SIZE_WINDOW_K * WINDOWS_PER_BLK;

  while (true)
  {
    uint32_t offset = (K.y - P.y) / 2;
    int2 Q = {K.x + (int)offset, K.y - (int)offset};

    bool Q_bottom_border = (Q.y == window_heigth) && window_bottom_border;
    bool Q_right_border = (Q.x == window_base) && window_right_border;
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

/**
 * @brief This kernel implements the merge algorithm given in the reference paper. It assumes that the problem has already been partitioned.
 * Inside each partition, the algorithm uses a moving window to perform the merge. The merge is performed as diagonal binary searches in shared memory.
 */
template <class T>
__global__ void mergeLarge_window_k(const T *A_ptr,
                                    const size_t A_size,
                                    const T *B_ptr,
                                    const size_t B_size,
                                    T *M_ptr,
                                    const int2 *Q_global)
{

  __shared__ T shared_mem_block[BLK_SIZE_WINDOW_K * WINDOWS_PER_BLK];

  int last_tid_of_block = BLK_SIZE_WINDOW_K - 1;

  //retrieve the corners of the current partition from global memory
  int2 Q_next = Q_global[blockIdx.x];
  int2 Q_prev = (blockIdx.x > 0) ? Q_global[blockIdx.x - 1] : make_int2(0, 0);

  //compute global position and size of current partition
  int x_start = Q_prev.x;
  int y_start = Q_prev.y;
  int base = Q_next.x - Q_prev.x;
  int height = Q_next.y - Q_prev.y;

  //shared memory loading
  T *A_block = shared_mem_block;
  T *B_block = shared_mem_block + height;

  for (unsigned i = threadIdx.x; i < height; i += BLK_SIZE_WINDOW_K) {
      A_block[i] = A_ptr[y_start + i];
  }

  // Load B_block
  for (unsigned i = threadIdx.x; i < base; i += BLK_SIZE_WINDOW_K) {
      B_block[i] = B_ptr[x_start + i];
  }
  __syncthreads();

  //keeps track of bottom-right corner of the current window
  __shared__ int2 Q_window;

  T *A_window = A_block;
  T *B_window = B_block;

  //keeps track of the remaining size of the current partition (each window may have a different size)
  int leftover_base = base, leftover_height = height;

  //loop over tiles
  for(int tile = 0; tile < WINDOWS_PER_BLK; tile++)
  {
    //index of the current diagonal in the block
    int block_diag_idx = tile * BLK_SIZE_WINDOW_K + threadIdx.x;

    //compute window size and whether it is on the border of the partition
    int window_base, window_heigth;
    bool window_bottom_border, window_right_border; 

    if(leftover_base <= BLK_SIZE_WINDOW_K)
    {
      window_base = leftover_base;
      window_right_border = true;
    }
    else
    {
      window_base = BLK_SIZE_WINDOW_K;
      window_right_border = false;
    }

    if(leftover_height <= BLK_SIZE_WINDOW_K)
    {
      window_heigth = leftover_height;
      window_bottom_border = true;
    }
    else
    {
      window_heigth = BLK_SIZE_WINDOW_K;
      window_bottom_border = false;
    }

    //define K and P for the current thread (in local block coordinates)
    int2 K, P;

    K.x = threadIdx.x <= window_heigth ? 0 : threadIdx.x - window_heigth;
    K.y = threadIdx.x <= window_heigth ? threadIdx.x : window_heigth;

    P.x = threadIdx.x <= window_base ? threadIdx.x : window_base;
    P.y = threadIdx.x <= window_base ? 0 : threadIdx.x - window_base;

    //perform binary search
    int2 Q_temp;
    if(block_diag_idx < height + base)
    {
      Q_temp = bin_search_window(A_window, B_window, K, P, M_ptr, window_base, window_heigth, window_bottom_border, window_right_border, block_diag_idx);
    }

    //last thread of the block broadcasts the coordinates of the starting point of the next window
    if(threadIdx.x == last_tid_of_block)
    {
      Q_window = Q_temp;
    }
    __syncthreads();

    //update pointers and sizes for the next window
    leftover_base -= Q_window.x;
    leftover_height -= Q_window.y;

    A_window += Q_window.y;
    B_window += Q_window.x;
  }

  return;  
}

/**
 * @brief This kernel improves the previous version. It starts with the same coarse partitioning (referred to as "boxes")
 * then each block performs a fine-grained partitioning in shared memory.
 * Finally the fine-grained partititions (tiles) are concurrently merged using a serial merge (one thread per tile).
 */
template <class T>
__global__ void mergeLarge_tiled_k(const T *A_ptr,
                                   const size_t A_size,
                                   const T *B_ptr,
                                   const size_t B_size,
                                   T *M_ptr,
                                   const int2 *Q_global)
{

  __shared__ T shared_mem_block[WORK_PER_BLK];

  //retrieve the corners of the current partition from global memory (coarse partitioning)
  int2 Q_next = Q_global[blockIdx.x];
  int2 Q_prev = (blockIdx.x > 0) ? Q_global[blockIdx.x - 1] : make_int2(0, 0);

  //compute global position and size of current partition
  unsigned x_start = Q_prev.x;
  unsigned y_start = Q_prev.y;
  unsigned base = Q_next.x - Q_prev.x;
  unsigned height = Q_next.y - Q_prev.y;

  //shared memory loading
  T *A_block = shared_mem_block;
  T *B_block = shared_mem_block + height;
  
  // Load A_block
  for (unsigned i = threadIdx.x; i < height; i += BLK_SIZE_TILED_K) {
      A_block[i] = A_ptr[y_start + i];
  }

  // Load B_block
  for (unsigned i = threadIdx.x; i < base; i += BLK_SIZE_TILED_K) {
      B_block[i] = B_ptr[x_start + i];
  }
  __syncthreads();

  //fine grained partitioning in shared memory
  int2 Q_start = fine_partition(A_block, height, B_block, base, WORK_PER_THREAD);

  T* A_window = A_block + Q_start.y;
  T* B_window = B_block + Q_start.x;

  //during serial merging, shared memory contains the local input arrays. To avoid allocating more shared memory, 
  //we temporarily store the merged array in a local variable. Once the merge is complete, we copy it back to shared memory and then to global memory.
  T M_tile[WORK_PER_THREAD];

  //perform serial tile merge
  unsigned i = 0, j = 0;
#pragma unroll
  //we don't check bounds, end of shared memory will contain garbage for threads that are not in bounds
  for(unsigned item = 0; item < WORK_PER_THREAD; item++)
  {
    //check whether to insert from A or B
    bool insert_A = (Q_start.y + i < height) && (Q_start.x + j >= base || A_window[i] < B_window[j]);

    if(insert_A)
    {
      M_tile[item] = A_window[i];
      i++;
    }
    else
    {
      M_tile[item] = B_window[j];
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
  unsigned effective_partition_size = base + height; //could be less than WORK_PER_BLK
  unsigned M_idx_start = Q_prev.x + Q_prev.y;
  for (unsigned item = threadIdx.x; item < effective_partition_size; item += BLK_SIZE_TILED_K)
  {
    M_ptr[M_idx_start + item] = shared_mem_block[item];
  }

  return;  
}