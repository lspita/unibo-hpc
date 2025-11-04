/****************************************************************************
 *
 * omp-pi.c - Monte Carlo approximation of PI
 *
 * Copyright (C) 2017--2025 Moreno Marzolla
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
% Monte Carlo approximation of $\pi$
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-09

The file [omp-pi.c](omp-pi.c) implements a serial Monte Carlo
algorithm for computing the approximate value of $\pi$. Monte Carlo
algorithms use pseudo-random numbers to evaluate some function of
interest.

![Figure 1: Monte Carlo computation of the value of $\pi$.](pi_Monte_Carlo.svg)

The idea is simple (see Figure 1). We generate $N$ random points
uniformly distributed over a square with corners at $(-1, -1)$ and
$(1, 1)$, and count the number $I$ of points falling inside the circle
with center $(0,0)$ and unitary radius. Then, we have:

$$
\frac{\text{N. of points inside the circle}}{\text{Total n. of points}} \approx
\frac{\text{Area of circle}}{\text{Area of enclosing square}}
$$

from which, substituting the appropriate variables:

$$
\frac{I}{N} \approx \frac{\pi}{4}
$$

hence $\pi \approx 4 I / N$. This estimate becomes more accurate as the
number of points $N$ increases.

The goal of this exercise is to modify the serial program to make use
of shared-memory parallelism using OpenMP.

## The hard (and inefficient) way

Start with a version that uses the `omp parallel` construct. Let $P$
be the number of OpenMP threads; then, the program operates as
follows:

1. The user specifies the number $N$ of points to generate as a
   command-line parameter, and the number $P$ of OpenMP threads using
   the `OMP_NUM_THREADS` environment variable.

2. Thread $p$ generates $N/P$ points using `generate_points()` and
   stores the result in `inside[p]`. `inside[]` is an integer array of
   length $P$ that must be declared outside the parallel region, since
   it must be shared across all OpenMP threads.

3. At the end of the parallel region, the master (thread 0) computes
   $I$ as the sum of the content of `inside[]`; from this the estimate
   of $\pi$ can be computed as above.

You may initially assume that the number of points $N$ is a multiple
of $P$; when you get a working program, relax this assumption to make
the computation correct for any value of $N$.

## The better way

A better approach is to let the compiler parallelize the "for" loop in
`generate_points()` using `omp parallel` and `omp for`.  There is a
problem, though: function `int rand(void)` is not thread-safe since it
modifies a global state variable, so it can not be called concurrently
by multiple threads. Instead, we use `int rand_r(unsigned int *seed)`
which is thread-safe but requires that each thread keeps a local
`seed`. We split the `omp parallel` and `omp for` directives, so that
a different local seed can be given to each thread like so:

```C
#pragma omp parallel default(none) shared(n, n_inside)
{
        const int my_id = omp_get_thread_num();
        \/\* Initialization of my_seed is arbitrary \*\/
        unsigned int my_seed = 17 + 19*my_id;
        ...
#pragma omp for reduction(+:n_inside)
        for (int i=0; i<n; i++) {
                \/\* call rand_r(&my_seed) here... \*\/
                ...
        }
        ...
}
```

Compile with:

        gcc -std=c99 -fopenmp -Wall -Wpedantic omp-pi.c -o omp-pi -lm

Run with:

        ./omp-pi [N]

For example, to compute the approximate value of $\pi$ using $P=4$
OpenMP threads and $N=20000$ points:

        OMP_NUM_THREADS=4 ./omp-pi 20000

## Files

- [omp-pi.c](omp-pi.c)

***/

/* The rand_r() function is available only if _XOPEN_SOURCE=600 */
#include <assert.h>
#include <stddef.h>
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include <math.h> /* for fabs */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_N_POINTS 10000000

typedef int (*generate_points_function_t)(int);

unsigned int random_seed(const int id) {
  // deterministic number sequences
  return 17 + (19 * id);
}

int generate_points_serial(int n) {
  puts("generate points: serial");
  unsigned int n_inside = 0;
  /* The C function `rand()` is not thread-safe, since it modifies a
     global seed; therefore, it can not be used inside a parallel
     region. We use `rand_r()` with an explicit per-thread
     seed. However, this means that in general the result computed
     by this program depends on the number of threads. */
  unsigned int my_seed = random_seed(omp_get_thread_num());
  for (int i = 0; i < n; i++) {
    /* Generate two random values in the range [-1, 1] */
    const double x = (2.0 * rand_r(&my_seed) / (double)RAND_MAX) - 1.0;
    const double y = (2.0 * rand_r(&my_seed) / (double)RAND_MAX) - 1.0;
    if ((x * x) + (y * y) <= 1.0) {
      n_inside++;
    }
  }
  return n_inside;
}

int omp_generate_points_parallel(int n) {
  puts("generate points: omp parallel");
  const int P = omp_get_max_threads();
  unsigned int* const inside = (unsigned int*)calloc(P, sizeof(unsigned int));

#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
    const int num_threads =
        omp_get_num_threads();  // get real number of threads
    const size_t start = (n * thread_id) / num_threads;
    const size_t end = (n * (thread_id + 1)) / num_threads;

    unsigned int n_inside = 0;
    unsigned int my_seed = random_seed(thread_id);
    for (size_t i = start; i < end; i++) {
      /* Generate two random values in the range [-1, 1] */
      const double x = (2.0 * rand_r(&my_seed) / (double)RAND_MAX) - 1.0;
      const double y = (2.0 * rand_r(&my_seed) / (double)RAND_MAX) - 1.0;
      if ((x * x) + (y * y) <= 1.0) {
        n_inside++;
      }
    }
    inside[thread_id] = n_inside;
  }
  unsigned int sum = 0;
  for (int i = 0; i < P; i++) {
    sum += inside[i];
  }
  free(inside);
  return sum;
}

int omp_generate_points_reduction(int n) {
  puts("generate points: omp reduction");
  unsigned int n_inside = 0;
#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
    unsigned int my_seed = random_seed(thread_id);
#pragma omp for reduction(+ : n_inside)
    for (int i = 0; i < n; i++) {
      /* Generate two random values in the range [-1, 1] */
      const double x = (2.0 * rand_r(&my_seed) / (double)RAND_MAX) - 1.0;
      const double y = (2.0 * rand_r(&my_seed) / (double)RAND_MAX) - 1.0;
      if ((x * x) + (y * y) <= 1.0) {
        n_inside++;
      }
    }
  }
  return n_inside;
}

int main(int argc, char* argv[]) {
  unsigned int n_points = DEFAULT_N_POINTS;
  unsigned int n_inside;
  const double PI_EXACT = 3.14159265358979323846;

  if (argc > 2) {
    fprintf(stderr, "Usage: %s [n_points]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (argc > 1) {
    n_points = atol(argv[1]);
  }

  generate_points_function_t generate_points_functions[] = {
      generate_points_serial,
      omp_generate_points_parallel,
      omp_generate_points_reduction,
  };
  const size_t generate_points_n =
      sizeof(generate_points_functions) / sizeof(generate_points_function_t);

  for (size_t i = 0; i < generate_points_n; i++) {
    puts("=== START ===");
    printf("Generating %u points...\n", n_points);
    const double tstart = omp_get_wtime();
    n_inside = generate_points_functions[i](n_points);
    const double elapsed = omp_get_wtime() - tstart;
    const double pi_approx = 4.0 * n_inside / (double)n_points;
    printf("PI approximation %f, exact %f, error %f%%\n", pi_approx, PI_EXACT,
           100.0 * fabs(pi_approx - PI_EXACT) / PI_EXACT);
    printf("Execution time %.3f\n", elapsed);
    puts("=== END ===");
  }

  return EXIT_SUCCESS;
}
