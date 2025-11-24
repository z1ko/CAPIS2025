// compute-sanitizer --leak-check full --tool racecheck

#include <cassert>

#define checkCudaErrors(Code) assert((Code) == cudaSuccess)
#define checkCudaLaunch(...) checkCudaErrors((__VA_ARGS__, cudaPeekAtLastError()))

static constexpr int warps = 2;
static constexpr int warp_size = 32;
static constexpr int threads = warps * warp_size;

__shared__ int smem_first[threads];
__shared__ int smem_second[warps];

__global__ void sum(int *data_in, int *sum_out)
{
    int tx = threadIdx.x;
    smem_first[tx] = data_in[tx] + tx;

    //__syncthreads();
    if (tx % warp_size == 0)
    {
        int wx = tx / warp_size;

        smem_second[wx] = 0;

        // Avoid loop unrolling for the purpose of racecheck demo
        #pragma unroll 1
        for (int i = 0; i < warp_size; ++i)
        {
            smem_second[wx] += smem_first[wx * warp_size + i];
        }
    }

    __syncthreads();

    if (tx == 0)
    {
        *sum_out = 0;
        for (int i = 0; i < warps; ++i)
        {
            *sum_out += smem_second[i];
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

    checkCudaErrors(cudaFree(data_in));
    checkCudaErrors(cudaFree(sum_out));
    return 0;
}
