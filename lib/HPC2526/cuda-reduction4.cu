/****************************************************************************
 *
 * cuda-reduction4.cu - Reduction with one CUDA block
 *
 * This version works for any array length n; however, it still
 * requires that BLKDIM is a power of two.
 *
 * Copyright (C) 2019, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ---------------------------------------------------------------------------
 *
 * This program realizes a on the GPU using a single thread
 * block. Each thread block copies a portion of the array in shared
 * memory; then, all threads within the same block cooperate to
 * compute the sum of the local data by organizing the computation as
 * a tree.
 *
 * Compile with:
 *
 *      nvcc cuda-reduction4.cu -o cuda-reduction4
 *
 * Run with:
 *
 *      ./cuda-reduction4
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* For this program to work, BLKDIM must be a power of two */
#define BLKDIM 1024

/* Note: *result must be initially zero for this kernel to work! */
__global__ void sum( int *a, int n, int *result )
{
    __shared__ int temp[BLKDIM];
    int lindex = threadIdx.x;
    int bsize = blockDim.x / 2;

    temp[lindex] = 0;

    for (int i = lindex; i < n; i += blockDim.x)
        temp[lindex] += a[i];

    /* wait for all threads to fill the shared memory */
    __syncthreads();

    /* All threads within the block cooperate to compute the local sum */
    while ( bsize > 0 ) {
        if ( lindex < bsize ) {
            temp[lindex] += temp[lindex + bsize];
        }
        bsize = bsize / 2;
        /* threads must synchronize before performing the next
           reduction step */
        __syncthreads();
    }

    if ( 0 == lindex ) {
        *result = temp[0];
    }
}

void init( int *v, int n )
{
    for (int i=0; i<n; i++) {
        v[i] = 2;
    }
}

int main( int argc, char *argv[] )
{
    int *h_a, result = 0;
    int *d_a, *d_result;
    int n = 1024*512;

    assert( (BLKDIM & (BLKDIM-1)) == 0 ); /* check if BLKDIM is a power of two using the "bit hack" from http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2 */

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    printf("Array length = %d\n", n);
    const size_t size = n * sizeof(*h_a);

    /* Allocate space for host copies of array */
    h_a = (int*)malloc(size);
    init(h_a, n);

    /* Allocate space for device copies of aarray */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_result, sizeof(*d_result));

    /* Copy inputs to device */
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    /* Copy the initial result (zero) to the device; this is important
       since the kernel requires *d_result to be initially zero. */
    cudaMemcpy(d_result, &result, sizeof(result), cudaMemcpyHostToDevice);

    /* Launch sum() kernel on the GPU */
    sum<<<1, BLKDIM>>>(d_a, n, d_result);

    /* Copy the result from device memory to host memory */
    cudaMemcpy(&result, d_result, sizeof(result), cudaMemcpyDeviceToHost);

    /* Check result */
    const int expected = 2*n;
    if ( result != expected ) {
        printf("Check FAILED: got %d, expected %d\n", result, expected);
    } else {
        printf("Check OK: sum = %d\n", result);
    }
    /* Cleanup */
    free(h_a);
    cudaFree(d_a); cudaFree(d_result);
    return EXIT_SUCCESS;
}
