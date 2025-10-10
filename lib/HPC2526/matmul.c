/****************************************************************************
 *
 * matmul.c - Dense matrix-matrix multiply showing caching effects
 *
 * Copyright (C) 2018, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * To compile the "plain" matrix-matrix multiply function:
 *
 *      gcc -std=c99 -Wall -Wpedantic matmul.c -o matmul-plain
 *
 * To compile the transposed matrix-matrix multiply function:
 *
 *      gcc -std=c99 -Wall -Wpedantic -DTRANSPOSE matmul.c -o matmul-transpose
 *
 * Run with:
 *
 *      ./matmul-plain [n] or ./matmul-transpose [n]
 *
 * where [n] is the matrix size.
 *
 * To see the number of cache misses:
 *
 *      perf stat -B -e task-clock,cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses ./matmul-plain
 *
 ****************************************************************************/

/* The following #define is required by the implementation of
   hpc_gettime(). It MUST be defined before including any other
   file. */
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>

#include "hpc.h"

/* Fills n x n square matrix m with random values */
void fill( double* m, int n )
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            m[i*n + j] = rand() / (double)RAND_MAX;
        }
    }
}

#ifdef TRANSPOSE

/* Cache-efficient computation of r = p * q, where p. q, r are n x n
   matrices. The caller is responsible for allocating the memory for
   r. This function allocates (and the frees) an additional n x n
   temporary matrix. */
void matmul( double *p, double* q, double *r, int n)
{
    double *qT = (double*)malloc( n * n * sizeof(double) );

    /* transpose q, storing the result in qT */
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            qT[j*n + i] = q[i*n + j];
        }
    }

    /* multiply p and qT row-wise */
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            double s = 0.0;
            for (int k=0; k<n; k++) {
                s += p[i*n + k] * qT[j*n + k];
            }
            r[i*n + j] = s;
        }
    }

    free(qT);
}

#else

/* compute r = p * q, where p, q, r are n x n matrices. The caller is
   responsible for allocating the memory for r */
void matmul( double *p, double* q, double *r, int n)
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            double s = 0.0;
            for (int k=0; k<n; k++) {
                s += p[i*n + k] * q[k*n + j];
            }
            r[i*n + j] = s;
        }
    }
}

#endif

int main( int argc, char *argv[] )
{
    int n = 1024;
    double tstart, tstop;
    double *p, *q, *r;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        n = atoi(argv[1]);
    }

    p = (double*)malloc( n * n * sizeof(double) );
    q = (double*)malloc( n * n * sizeof(double) );
    r = (double*)malloc( n * n * sizeof(double) );

    fill(p, n);
    fill(q, n);

    /* You should take multiple measurements and average them; here we
       take just one for simplicity. */
#ifdef TRANSPOSE
    printf("Starting transpose matrix-matrix multiply (n=%d)...", n);
#else
    printf("Starting plain matrix-matrix multiply (n=%d)...", n);
#endif

    tstart = hpc_gettime();
    matmul(p, q, r, n);
    tstop = hpc_gettime();
    printf(" done\n\nr[0][0] = %f\n\telapsed time = %.2f s\n\n", r[0], tstop - tstart);

    free(p);
    free(q);
    free(r);

    return EXIT_SUCCESS;
}
