#pragma once
#include <vector>
typedef int2 coordinate;

#define THREAD_NUM (64)

template <class T>
std::vector<T> mergeSmall_k_cpu(std::vector<T> vector_1, std::vector<T> vector_2)
{
    static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");

    size_t v_size_1 = vector_1.size();
    size_t v_size_2 = vector_2.size();
    size_t m_size = v_size_1 + v_size_2;
    std::vector<T> merged_vector(m_size);

    size_t idx_i = 0, idx_j = 0;

    while (idx_i + idx_j < m_size)
    {
        if (idx_i >= v_size_1)
        {
            merged_vector[idx_i + idx_j] = vector_2[idx_j];
            idx_j++;
        }
        else if (idx_j >= v_size_2 || vector_1[idx_i] < vector_2[idx_j])
        {
            merged_vector[idx_i + idx_j] = vector_1[idx_i];
            idx_i++;
        }
        else
        {
            merged_vector[idx_i + idx_j] = vector_2[idx_j];
            idx_j++;
        }
    }
    return merged_vector;
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
__global__ void mergeSmall_k1(const T *v_1_ptr,
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
__global__ void mergeSmall_k2(const T *v_1_ptr,
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

    __shared__ int2 Q_shift;
    __shared__ T A[THREAD_NUM + 1];
    __shared__ T B[THREAD_NUM + 1];
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
            Q_shift.y += THREAD_NUM - 1 - threadIdx.x;
        else
            Q_shift.x += THREAD_NUM - 1 - threadIdx.x;
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
