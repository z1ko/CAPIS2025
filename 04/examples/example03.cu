#include <cassert>

#define checkCudaErrors(Code) assert((Code) == cudaSuccess)
#define checkCudaLaunch(...) checkCudaErrors((__VA_ARGS__, cudaPeekAtLastError()))

static constexpr int threads = 128;

__shared__ int smem[threads];

__global__ void sum(int *data_in, int *sum_out)
{
    int tx = threadIdx.x;
    smem[tx] = data_in[tx] + tx;

    if (tx == 0)
    {
        *sum_out = 0;

        // Avoid loop unrolling for the purpose of racecheck demo
        #pragma unroll 1
        for (int i = 0; i < threads; ++i)
        {
            *sum_out += smem[i];
        }
    }
}

int main()
{
    int *data_in = nullptr;
    int *sum_out = nullptr;

    checkCudaErrors(cudaMalloc((void**)&data_in, sizeof(int) * threads));
    checkCudaErrors(cudaMalloc((void**)&sum_out, sizeof(int)));
    checkCudaErrors(cudaMemset(data_in, 0, sizeof(int) * threads));

    checkCudaLaunch(sum<<<1, threads>>>(data_in, sum_out));
    checkCudaErrors(cudaDeviceSynchronize());
    
    return 0;
}