/****************************************************************************
 *
 * omp-matmul.c - Dense matrix-matrix multiply
 *
 * Copyright (C) 2017, 2022, 2025 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 *
 *      gcc -fopenmp -std=c99 -Wall -Wpedantic omp-matmul.c -o omp-matmul
 *
 * Run with:
 *
 *      ./omp-matmul [n]
 *
 * (n = matrix size).
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Fills matrix m of size n x n with random values. */
void fill( float* m, int n )
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            m[i*n + j] = rand() / (double)RAND_MAX; /* RAND_MAX can not be case to float without loss of precision */
        }
    }
}

/* Cache-efficient computation of r = p * q, where p. q, r are n x n
   matrices. The caller is responsible for allocating memory for
   r. This function allocates (and the frees) an additional n x n
   temporary matrix qT. */
void matmul_transpose( float *p, float* q, float *r, int n)
{
    float *qT = (float*)malloc( n * n * sizeof(float) );

    /* transpose q, storing the result in qT */
#pragma omp parallel
    {
#pragma omp for
        for (int  i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                qT[j*n + i] = q[i*n + j];
            }
        }

        /* multiply p and qT row-wise */
#pragma omp for schedule(runtime)
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                float s = 0.0;
                for (int k=0; k<n; k++) {
                    s += p[i*n + k] * qT[j*n + k];
                }
                r[i*n + j] = s;
            }
        }
    }

    free(qT);
}

int main( int argc, char *argv[] )
{
    int n = 4096;
    float *p, *q, *r;

    if ( argc > 2 ) {
        printf("Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }

    p = (float*)malloc( n * n * sizeof(float) );
    q = (float*)malloc( n * n * sizeof(float) );
    r = (float*)malloc( n * n * sizeof(float) );

    fill(p, n);
    fill(q, n);

    printf("Matrix-Matrix multiplication (%dx%d)\n", n, n);

    const double tstart = omp_get_wtime();
    matmul_transpose(p, q, r, n);
    const double elapsed = omp_get_wtime() - tstart;

    printf("Execution time %f\n", elapsed);

    free(p);
    free(q);
    free(r);

    return EXIT_SUCCESS;
}
