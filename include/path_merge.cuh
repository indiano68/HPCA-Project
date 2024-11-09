#pragma once
#include <vector>
typedef int2 coordinate;
// struct coordinate
// {
//     int32_t x;
//     int32_t y;
// };

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
