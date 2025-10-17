/****************************************************************************
 *
 * omp-dot.c - Dot product
 *
 * Copyright (C) 2018--2025 Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/***
% Dot product
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-09

The file [omp-dot.c](omp-dot.c) contains a serial program that
computes the dot product of two arrays `v1[]` and `v2[]`. The program
receives the array lengths $n$ as the only parameter on the command
line. The arrays are initialized deterministically, so that their
scalar product is known awithout computing it explicitly.  The dot
product of `v1[]` and `v2[]` is defined as:

$$
\sum_{i = 0}^{n-1} v1[i] \times v2[i]
$$

The goal of this exercise is to parallelize the serial program using
the `omp parallel` construct with the appropriate clauses. It is
instructive to begin without using the `omp parallel for` directive
and computing the endpoints of the iterations explicitly.  To this
aim, let $P$ be the size of the OpenMP team; partition the
arrays into $P$ blocks of approximately uniform size. Thread $p$ ($0
\leq p < P$) computes the dot product `my_p` of the subvectors with
indices $\texttt{my_start}, \ldots, \texttt {my_end}-1$:

$$
\texttt{my_p}: = \sum_{i=\texttt{my_start}}^{\texttt{my_end}-1} v1[i] \times
v2[i]
$$

There are several ways to accumulate partial results. One possibility
is to store the value computed by thread $p$ on `partial_p[p]`, where
`partial_p[]` is an array of length $P$. In this way each thread
writes on different elements of `partial_p[]` and no race conditions
are possible. After the parallel region completes, the master thread
computes the final result by summing the content of `partial_p[]`. Be
sure to handle the case where $n$ is not an integer multiple of $P$
correctly.

The solution above is instructive but tedious and inefficient.  Unless
there are specific reasons to do so, in practice you should use the
`omp parallel for` directive with the `reduction()` clause, and let
the compiler take care of everything.

To compile:

        gcc -fopenmp -std=c99 -Wall -Wpedantic omp-dot.c -o omp-dot

To execute:

        ./omp-dot [n]

For example, if you want to use two OpenMP threads:

        OMP_NUM_THREADS=2 ./omp-dot 1000000

## File

- [omp-dot.c](omp-dot.c)

***/

#include <assert.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_MATRIX_SIZE ((long)(10 * 1024) * 1024L)
#define MAX_MATRIX_SIZE ((long)(512 * 1024) * 1024L)

typedef int (*dot_function_t)(const int*, const int*, size_t);

void fill(int* v1, int* v2, size_t n) {
  const int seq1[3] = {3, 7, 18};
  const int seq2[3] = {12, 0, -2};
  for (size_t i = 0; i < n; i++) {
    v1[i] = seq1[i % 3];
    v2[i] = seq2[i % 3];
  }
}

int dot(const int* v1, const int* v2, size_t n) {
  puts("dot product: serial");
  int result = 0;
  for (size_t i = 0; i < n; i++) {
    result += v1[i] * v2[i];
  }
  return result;
}

int dot_parallel(const int* v1, const int* v2, const size_t n) {
  puts("dot product: parallel standard");
  int result = 0;
#pragma omp parallel default(none) shared(n, v1, v2) reduction(+ : result)
  {
    const int id = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();
    const size_t start = (n * id) / num_threads;
    const size_t end = (n * (id + 1)) / num_threads;
    for (size_t i = start; i < end; i++) {
      result += v1[i] * v2[i];
    }
  }
  return result;
}

int dot_parallel_for(const int* v1, const int* v2, const size_t n) {
  puts("dot product: parallel for");
  int result = 0;
#pragma omp parallel for default(none) shared(n, v1, v2) reduction(+ : result)
  for (size_t i = 0; i < n; i++) {
    result += v1[i] * v2[i];
  }
  return result;
}

int main(int argc, char* argv[]) {
  int* v1;
  int* v2;

  if (argc > 2) {
    fprintf(stderr, "Usage: %s [n]\n", argv[0]);
    return EXIT_FAILURE;
  }

  size_t n = DEFAULT_MATRIX_SIZE; /* array length */
  if (argc > 1) {
    n = atol(argv[1]);
  }

  if (n > MAX_MATRIX_SIZE) {
    fprintf(stderr,
            "FATAL: Array too long (requested length=%lu, maximum length=%lu\n",
            (unsigned long)n, (unsigned long)MAX_MATRIX_SIZE);
    return EXIT_FAILURE;
  }

  const int expect = (n % 3 == 0 ? 0 : 36);
  dot_function_t dot_functions[] = {
      dot,
      dot_parallel,
      dot_parallel_for,
  };
  const size_t dot_functions_n = sizeof(dot_functions) / sizeof(dot_function_t);

  for (size_t i = 0; i < dot_functions_n; i++) {
    puts("=== START ===");
    printf("Initializing array of length %lu\n", (unsigned long)n);
    v1 = (int*)malloc(n * sizeof(v1[0]));
    assert(v1 != NULL);
    v2 = (int*)malloc(n * sizeof(v2[0]));
    assert(v2 != NULL);
    fill(v1, v2, n);
    const double tstart = omp_get_wtime();
    const int result = dot_functions[i](v1, v2, n);
    const double elapsed = omp_get_wtime() - tstart;
    if (result == expect) {
      printf("Test OK\n");
    } else {
      printf("Test FAILED: expected %d, got %d\n", expect, result);
    }
    printf("Execution time %.3f\n", elapsed);
    free(v1);
    free(v2);
    puts("=== END ===");
  }

  return EXIT_SUCCESS;
}
