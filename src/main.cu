/****************************************************************************
 *
 * cuda-hello0.cu - Hello world with CUDA (no device code)
 *
 * Based on the examples from in the CUDA toolkit documentation
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * Last updated in 2017 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 *
 *      nvcc cuda-hello0.cu -o cuda-hello0
 *
 * Run with:
 *
 *      ./cuda-hello0
 *
 ****************************************************************************/

#include <stdint.h>
#include <stdio.h>

__global__ void ker() {
  const uint32_t id = threadIdx.x;
  printf("Thread ID: %u\n", id);
}

int main(void) {
  printf("Hello, world!\n");
  ker<<<1, 1>>>();

  const cudaError_t err = cudaDeviceSynchronize();
  if (err == cudaSuccess) {
    printf("CUDA: success\n");
  } else {
    const int32_t _ =
        fprintf(stderr, "CUDA: error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  return 0;
}
