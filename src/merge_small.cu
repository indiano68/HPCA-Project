__global__ void mergeSmall_k(int* A, int* B, int* M, int sizeA, int sizeB)
{
    __shared__ int sharedA[512];
    __shared__ int sharedB[512];

    // Load A and B into shared memory
    int tid = threadIdx.x;
    if (tid < sizeA) sharedA[tid] = A[tid];
    if (tid < sizeB) sharedB[tid] = B[tid];
    __syncthreads();

    // Each thread handles one diagonal of the merge path
    int k = threadIdx.x;
    if (k < sizeA + sizeB)
    {
        int start = max(0, k - sizeB);
        int end = min(k, sizeA);
        
        // Binary search to find the intersection point
        while (start <= end)
        {
            int i = (start + end) / 2;
            int j = k - i;
            
            if (i < sizeA && j > 0 && sharedB[j-1] > sharedA[i])
                start = i + 1;
            else if (j < sizeB && i > 0 && sharedA[i-1] > sharedB[j])
                end = i - 1;
            else
            {
                // Found the intersection, write to output
                if (j >= sizeB || (i < sizeA && sharedA[i] <= sharedB[j]))
                    M[k] = sharedA[i];
                else
                    M[k] = sharedB[j];
                break;
            }
        }
    }
}