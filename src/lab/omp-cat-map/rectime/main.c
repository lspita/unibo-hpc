/****************************************************************************
 *
 * cat-map-rectime.c - Minimum recurrence time of Arnold's  cat map
 *
 * Copyright (C) 2017--2021, 2024, 2025 Moreno Marzolla
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
% Minimum Recurrence Time of Arnold's cat map
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-09

This program computes the _Minimum Recurrence Time_ of Arnold's cat
map for an image of given size $N \times N$. The minimum recurrence
time is the minimum number of iterations of Arnold's cat map that
return back the original image.

The minimum recurrence time depends on the image size $n$, but no
simple relation is known. Table 1 shows the minimum recurrence time
for some values of $N$.

:Table 1: Minimum recurrence time for some image sizes $N$

    $N$   Minimum recurrence time
------- -------------------------
     64                        48
    128                        96
    256                       192
    512                       384
   1368                        36
------- -------------------------

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-cat-map-rectime.c -o
omp-cat-map-rectime

Run with:

        ./omp-cat-map-rectime [N]

Example:

        ./omp-cat-map-rectime 1024

## Files

- [omp-cat-map-rectime.c](omp-cat-map-rectime.c)

***/

#include <assert.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* Compute the Greatest Common Divisor (GCD) of integers a>0 and b>0 */
int gcd(int a, int b) {
  assert(a > 0);
  assert(b > 0);

  while (b != a) {
    if (a > b) {
      a = a - b;
    } else {
      b = b - a;
    }
  }
  return a;
}

/* compute the Least Common Multiple (LCM) of integers a>0 and b>0 */
int lcm(int a, int b) {
  assert(a > 0);
  assert(b > 0);
  return (a / gcd(a, b)) * b;
}

/**
 * Compute the recurrence time of Arnold's cat map applied to an image
 * of size (n*n). For each point (x,y), compute the minimum recurrence
 * time k(x,y). The minimum recurrence time for the whole image is the
 * Least Common Multiple of all k(x,y).
 */
int cat_map_rectime(int n) {
  /* [TODO] Implement this function; start with a working serial
     version, then parallelize. */

  if (n == 0) return 0;

  const int total_pixels = n * n;
  int* const recurrences = (int*)calloc(total_pixels, sizeof(int));

#pragma omp parallel for collapse(2) default(none) shared(n, recurrences)
  for (int y = 0; y < n; y++) {
    for (int x = 0; x < n; x++) {
      int xcur = x, ycur = y;
      do {
        const int xnext = (2 * xcur + ycur) % n;
        const int ynext = (xcur + ycur) % n;
        xcur = xnext;
        ycur = ynext;
        recurrences[y * n + x]++;
      } while (xcur != x || ycur != y);
    }
  }

// custom reduction
// slower when parallelized ????
#pragma omp parallel default(none) shared(recurrences, total_pixels)
  {
    int n1 = total_pixels;
    int n2;
    do {
      n2 = (n1 + 1) / 2;
#pragma omp for
      for (int i = 0; i < n2; i++) {
        if (i + n2 < n1) {
          recurrences[i] = lcm(recurrences[i], recurrences[i + n2]);
        }
      }
      n1 = n2;
    } while (n2 > 1);
  }

  const int total_lcm = recurrences[0];
  free(recurrences);

  return total_lcm;
}

int main(int argc, char* argv[]) {
  int n, k;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s image_size\n", argv[0]);
    return EXIT_FAILURE;
  }
  n = atoi(argv[1]);
  const double tstart = omp_get_wtime();
  k = cat_map_rectime(n);
  const double elapsed = omp_get_wtime() - tstart;
  printf("%d\n", k);

  printf("Execution time %.3f\n", elapsed);

  return EXIT_SUCCESS;
}
