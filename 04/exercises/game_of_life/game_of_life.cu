#define PPROG_IMPLEMENTATION

#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

#include <chrono>
#include <thread>

#include <raylib.h>

// ===================================================================================
// CONFIGURATION

#define TILE_SIZE_X 32
#define TILE_SIZE_Y 32

#define WIDTH  640
#define HEIGHT 360

// Set to 1 to use the shmem kernel
#define USE_SHMEM 0

// ===================================================================================
// UTILITIES

// Access the matrix by indices
#define at(i, j) (((i) * WIDTH + (j)))

// Get value in the matrix
#define matrix_get(matrix, i, j) ((at(i, j) <= WIDTH * HEIGHT - 1) ? matrix[at(i, j)] : 0)

// Check cuda errors
#define cuda_check_error()                                                     \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

// ===================================================================================
// @@ IMPLEMENTATION

//@@ Write the stencil kernel without shared memory
__global__ void stencil_gpu_naive(int* current, int* next) {
    // @@ ...
}

//@@ Write the stencil kernel with shared memory
__global__ void stencil_gpu_shmem(int* current, int* next) {
    // @@ ...
}

// ===================================================================================

void update_gpu(int* d_current, int* d_next, int* h_current) {

    dim3 block(TILE_SIZE_X, TILE_SIZE_Y);
    dim3 grid((WIDTH + TILE_SIZE_X - 1) / TILE_SIZE_X, (HEIGHT + TILE_SIZE_Y - 1) / TILE_SIZE_Y);

    auto frame_timer = timerGPU{};
    frame_timer.start();

#if USE_SHMEM == 0
    stencil_gpu_naive<<<grid, block>>>(d_current, d_next);
#endif

#if USE_SHMEM == 1
    stencil_gpu_shmem<<<grid, block>>>(d_current, d_next);
#endif

    frame_timer.stop();
    printf("frame time gpu: %f ms\n", frame_timer.elapsed_ms());

    // Store result back to CPU
    cudaMemcpy(h_current, d_next, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_current, d_next, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToDevice);
}

int main(int argc, char** argv) {

    // Generates random matrix data
    printf("generating random data...\n");
    auto h_current = random_matrix<int>(WIDTH, HEIGHT, 2043 /* SEED */);
    printf("done\n");

    // Allocate matrix in GPU memory
    int *d_current;
    cudaMalloc((void**)&d_current, h_current.size() * sizeof(int));
    cudaMemcpy(d_current, h_current.data(), h_current.size() * sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error();

    // Create result matrix
    int *d_next;
    cudaMalloc(&d_next, h_current.size() * sizeof(int));
    cuda_check_error();

    InitWindow(1280, 720, "Game of Life");
    SetTargetFPS(60);

    Image image = GenImageColor(WIDTH, HEIGHT, WHITE);
    Texture2D render_target = LoadTextureFromImage(image);

    while (!WindowShouldClose()) {

        // Update state
        update_gpu(d_current, d_next, h_current.data());

        // Update texture
        for (int i = 0; i < WIDTH * HEIGHT; ++i)
            ((Color*)image.data)[i] = h_current[i] ? WHITE : BLACK; 

        UpdateTexture(render_target, image.data);

        BeginDrawing();
        ClearBackground(BLACK);

        //DrawTexture(render_target, 0, 0, WHITE);
        DrawTextureEx(render_target, {0, 0}, 0.0f, 2.0f, WHITE);
		
		EndDrawing();
	}

    // Free cuda memory
    //cudaFree(d_matrix);
    //cudaFree(d_result);

    CloseWindow();
    return 0;
}
