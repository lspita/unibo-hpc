/****************************************************************************
 *
 * mpi-pi.c - Monte Carlo approximatino of PI
 *
 * Copyright (C) 2017--2022, 2024 Moreno Marzolla
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
% Monte Carlo approximation of PI
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-10-25

The file [mpi-pi.c](mpi-pi.c) contains a serial program for computing
the approximate value of $\pi$ using a Monte Carlo algorithm. Monte
Carlo algorithms use pseudorandom numbers to compute an approximation
of some quantity of interest.

![Figure 1: Monte Carlo computation of the value of $\pi$](pi_Monte_Carlo.svg)

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

Modify the serial program to parallelize the computation. Several
parallelization strategies are possible, but for now you are advised
to implement the following one ($P$ is the number of MPI processes):

1. Each process gets the value of the number of points $N$ from the
   command line. You may initially assume that $N$ is a multiple of
   $P$, and then relax this requirement to make the program with any
   value of $N$.

2. Each process $p$, including the master, generates $N/P$ random
   points and keeps track of the number $I_p$ of points that fall
   inside the circle;

3. The master computes the total number $I$ of points that fall inside
   the circle as the sum of $I_p$, $p=0, \ldots, P-1$.

Step 3 involves a reduction operation. Start by implementing the
inefficient solution, e.g., each process $p > 0$ sends its local value
$I_p$ to the master using point-to-point send/receive operations. The
master receives $I_p$ from all each process $p = 1, \ldots, P-1$ (the
master already knows $I_0$), computes their sum $I$ and the prints the
approximate value of $\pi$ as $(4 I / N)$.

Once you have a working implementation, modify it to use the preferred
solution, i.e., `MPI_Reduce()` instead of point-to-point
communications.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-pi.c -o mpi-pi -lm

To execute:

        mpirun -n P ./mpi-pi [N]

Example, using 4 MPI processes:

        mpirun -n 4 ./mpi-pi 1000000

## Files

- [mpi-pi.c](mpi-pi.c)

***/
#include <math.h> /* for fabs() */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <time.h>   /* for time() */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MASTER_RANK 0
#define VALUE_TAG 0

typedef int (*generate_points_function_t)(int);

int my_rank, comm_sz;

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1 */
int generate_points(int n) {
  int n_inside = 0;
  for (int i = 0; i < n; i++) {
    const double x = (rand() / (double)RAND_MAX * 2.0) - 1.0;
    const double y = (rand() / (double)RAND_MAX * 2.0) - 1.0;
    if (x * x + y * y < 1.0) {
      n_inside++;
    }
  }
  return n_inside;
}

int mpi_generate_points_naive(int n) {
  int I;
  switch (my_rank) {
    case MASTER_RANK:
      puts("generate points: mpi naive");
      I = generate_points((n / comm_sz) + (n % comm_sz));
      for (int rank = 0; rank < comm_sz - 1; rank++) {
        int buff;
        MPI_Recv(&buff, 1, MPI_INT, MPI_ANY_SOURCE, VALUE_TAG, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        I += buff;
      }
      break;
    default:
      I = generate_points(n / comm_sz);
      MPI_Send(&I, 1, MPI_INT, MASTER_RANK, VALUE_TAG, MPI_COMM_WORLD);
      break;
  }
  return I;
}

int main(int argc, char* argv[]) {
  int inside = 0, npoints = 1000000;
  double pi_approx;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  if (argc > 1) {
    npoints = atoi(argv[1]);
  }

  /* Each process initializes the pseudo-random number generator; if
     we don't do this (or something similar), each process would
     produce the exact same sequence of pseudo-random numbers! */
  srand(my_rank * 11 + 7);

  /* [TODO] This is not a true parallel version; the master does
     everything */
  generate_points_function_t generate_points_functions[] = {
      mpi_generate_points_naive,
  };
  const size_t generate_points_n =
      sizeof(generate_points_functions) / sizeof(generate_points_function_t);
  for (size_t i = 0; i < generate_points_n; i++) {
    if (my_rank == MASTER_RANK) {
      puts("=== START ===");
      printf("Generating %u points...\n", npoints);
    }
    inside = generate_points_functions[i](npoints);
    if (my_rank == MASTER_RANK) {
      pi_approx = 4.0 * inside / (double)npoints;
      printf("PI approximation is %f (true value=%f, rel error=%.3f%%)\n",
             pi_approx, M_PI, 100.0 * fabs(pi_approx - M_PI) / M_PI);
      puts("=== END ===");
    }
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
