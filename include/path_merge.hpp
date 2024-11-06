#pragma once 
#include<vector>



template <class T>
std::vector<T> path_merge(std::vector<T> vector_1, std::vector<T> vector_2)
{
    static_assert(std::is_arithmetic<T>::value, "Template type must be numeric");

    size_t v_size_1 = vector_1.size();
    size_t v_size_2 = vector_2.size();
    size_t m_size = v_size_1+ v_size_2;
    std::vector<T> merged_vector(m_size);

    size_t idx_i = 0, idx_j = 0;

    while( idx_i + idx_j < m_size)
    {
        if(idx_i >= v_size_1)
        {
            merged_vector[idx_i+idx_j] = vector_2[idx_j];
            idx_j++;
        }
        else if( idx_j>= v_size_2 || vector_1[idx_i] < vector_2[idx_j] )
        {
            merged_vector[idx_i+idx_j] = vector_1[idx_i];
            idx_i++;
        }
        else
        {
            merged_vector[idx_i+idx_j] = vector_2[idx_j];
            idx_j++;
        }

    }
    return merged_vector;
}
