#include <assert.h>

template <class T>
__device__ inline int2 merge_tile(const T *v_A_ptr,
                                  const T *v_B_ptr,
                                  T *v_out_ptr,
                                  const size_t tile_size,
                                  const bool shift)
{
    assert(tile_size == blockDim.x);
    int thread_idx = threadIdx.x;
    if (shift)
        thread_idx++;
    int2 K = {0, 0};
    int2 P = {0, 0};
    int2 Q = {0, 0};
    K.y = thread_idx;
    P.x = thread_idx;

    while (true)
    {
        size_t offset = abs((K.y - P.y)) / 2;
        Q.x = K.x + offset;
        Q.y = K.y - offset;
        if (Q.y == tile_size || Q.x == 0 || v_A_ptr[Q.y] > v_B_ptr[Q.x - 1])
        {
            if (Q.x == tile_size || Q.y == 0 || v_A_ptr[Q.y - 1] <= v_B_ptr[Q.x])
            {
                if (Q.y < tile_size && (Q.x == tile_size || v_A_ptr[Q.y] <= v_B_ptr[Q.x]))
                {
                    v_out_ptr[threadIdx.x] = v_A_ptr[Q.y];
                }
                else
                {
                    v_out_ptr[threadIdx.x] = v_B_ptr[Q.x];
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
    return Q;
}

template <class T>
__global__ void merge_tile_k(const T *v_A_ptr,
                             const T *v_B_ptr,
                             T *v_out_ptr,
                             const size_t tile_size,
                             const bool shift)
{
    merge_tile(v_A_ptr, v_B_ptr, v_out_ptr, tile_size, shift);
}

template <class T>
__global__ void merge_small_tiled_k(const T *v_A_ptr,
                                    const size_t v_A_size,
                                    const T *v_B_ptr,
                                    const size_t v_B_size,
                                    T *v_out_ptr,
                                    const size_t v_out_size)
{
    extern __shared__ uint8_t total_shared_buffer[];
    T *tile_A   = (T *)total_shared_buffer;
    T *tile_B   = (T *)total_shared_buffer + blockDim.x;
    T *tile_out = (T *)total_shared_buffer + blockDim.x + blockDim.x;
    int2 *Q_shift = (int2 *)((T*)total_shared_buffer + blockDim.x + blockDim.x + blockDim.x);
    int2 Q_shift_buffer;
    const T *current_A_ptr = v_A_ptr;
    const T *current_B_ptr = v_B_ptr;
    int current_A_idx = 0;
    int current_B_idx = 0;
    int current_out_idx = 0;
    T *current_out_ptr = v_out_ptr;
    bool shift = false;
    int tot_tiles = (v_out_size-1) / (blockDim.x-1);
    T padder = max(v_A_ptr[v_A_size-1],v_B_ptr[v_B_size-1]);

    for (int n_tile = 0; n_tile < tot_tiles; n_tile++, shift = false)
    {    
        if(current_A_idx+threadIdx.x < v_A_size)
            tile_A[threadIdx.x] = current_A_ptr[threadIdx.x];
        else
            tile_A[threadIdx.x] = padder;

        if(current_B_idx+threadIdx.x < v_B_size)
            tile_B[threadIdx.x] = current_B_ptr[threadIdx.x];
        else
            tile_B[threadIdx.x] = padder;

        __syncthreads();
        Q_shift_buffer = merge_tile(tile_A, tile_B, tile_out, blockDim.x, shift);
        if (threadIdx.x == blockDim.x - 1)
            *Q_shift = Q_shift_buffer;
        __syncthreads();
        current_out_ptr[threadIdx.x] = tile_out[threadIdx.x];
        current_A_ptr+=Q_shift->y;
        current_B_ptr+=Q_shift->x;
        current_A_idx+=Q_shift->y;
        current_B_idx+=Q_shift->x;
        current_out_idx+= blockDim.x-1;
        current_out_ptr+= blockDim.x-1;
    }
    
    if(current_out_idx < v_out_size)
    {
        if(current_A_idx+threadIdx.x < v_A_size)
            tile_A[threadIdx.x] = current_A_ptr[threadIdx.x];
        else
            tile_A[threadIdx.x] = padder;

        if(current_B_idx+threadIdx.x < v_B_size)
            tile_B[threadIdx.x] = current_B_ptr[threadIdx.x];
        else
            tile_B[threadIdx.x] = padder;
        __syncthreads();
        merge_tile(tile_A, tile_B, tile_out, blockDim.x, shift);
        if(current_out_idx +threadIdx.x < v_out_size)
            current_out_ptr[threadIdx.x] = tile_out[threadIdx.x];
    }
}
