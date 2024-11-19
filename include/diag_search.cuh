#pragma once
#define DEBUG false

template <class T>
__device__ void print_shared(T *A_shared, T *B_shared, int base, int height)
{
  if(threadIdx.x == 0)
  {
    for(int i = 0; i < height + 2; i++)
    {
      printf("block %d A[%d] = %d\n", blockIdx.x, i, A_shared[i]);
    }
    for(int i = 0; i < base + 2; i++)
    {
      printf("block %d B[%d] = %d\n", blockIdx.x, i, B_shared[i]);
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
__device__ void block_bin_search(const T *A_local, const T *B_local, int2 K, int2 P, T *M_global, bool blk_left_border, bool blk_right_border, bool blk_top_border, bool blk_bottom_border, int base, int height)
{

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (true)
  {
    uint32_t offset = abs(K.y - P.y) / 2;
    int2 Q = {K.x + (int)offset, K.y - (int)offset};

    bool Q_bottom_border = (blk_bottom_border && Q.y == height);
    bool Q_left_border = (blk_left_border && Q.x == 0);
    bool Q_right_border = (blk_right_border && Q.x == base);
    bool Q_top_border = (blk_top_border && Q.y == 0);

    if (Q_bottom_border || Q_left_border || A_local[Q.y] > B_local[Q.x - 1])
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