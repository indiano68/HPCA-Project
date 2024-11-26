#pragma once
#include <config.h> // THREADS_PER_BLOCK, TILE_SIZE, ...

template <class T>
__device__ void print_shared(T *A_shared, T *B_shared, int base, int height)
{
  if(threadIdx.x == 0)
  {
    // for(int i = 0; i < height + 2; i++)
    // {
    //   printf("block %d A[%d] = %d\n", blockIdx.x, i, A_shared[i]);
    // }
    // for(int i = 0; i < base + 2; i++)
    // {
    //   printf("block %d B[%d] = %d\n", blockIdx.x, i, B_shared[i]);
    // }
    for(int i = 0; i < height; i++)
    {
      printf("block %d A_local[%d] = %d\n", blockIdx.x, i, A_shared[i]);
    }
    for(int i = 0; i < base; i++)
    {
      printf("block %d B_local[%d] = %d\n", blockIdx.x, i, B_shared[i]);
    }
  }
}

template <class T>
__device__ __forceinline__ int2 explorative_search(const T *A_ptr, const size_t A_size, const T *B_ptr, const size_t B_size, int2 K, int2 P)
{

  while (true)
  {
    //uint32_t offset = abs(K.y - P.y) / 2; don't need abs because we know that K.y > P.y
    uint32_t offset = (K.y - P.y) / 2;
    int2 Q = {K.x + (int)offset, K.y - (int)offset};

    // i primi due if detectano se il punto è plausibile
    if (Q.y == A_size || Q.x == 0 || A_ptr[Q.y] > B_ptr[Q.x - 1])
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
__device__ __forceinline__ int2 block_bin_search_tiled(const T *A_local, const T *B_local, int2 K, int2 P, T *M_global, int tile_base, int tile_height, bool tile_bottom_border, bool tile_right_border, int block_diag_idx)
{

  int M_idx = block_diag_idx + blockIdx.x * TILE_SIZE * TILES_PER_BLOCK;

  while (true)
  {
    //uint32_t offset = abs(K.y - P.y) / 2; don't need abs because we know that K.y > P.y
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
