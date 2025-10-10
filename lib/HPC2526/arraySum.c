/****************************************************************************
 *
 * arraySum.c -- sequential and parallel array sum
 *
 * Written in 2023 by Gianluigi Zavattaro
 * Modified in 2025 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all
 * copyright and related and neighboring rights to this software to the
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see
 * <http://creativecommons.org/publicdomain/zero/1.0/>.
 *
 * --------------------------------------------------------------------------
 *
 * Sequential and parallel solutions (some are WRONG) to the problem
 * of summing the elements in an array.
 *
 * Compile with:
 *
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp arraySum.c -o arraySum
 *
 * Execute with an appropriate number of threads, e.g.
 *
 *      OMP_NUM_THREADS=6 ./arraySum 2000000
 *
 * (the optional parameter indicates the array length)
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Sequential version. */
long sum_seq(const int *v, int n)
{
    int i;
    long sum = 0;

    printf("*** SEQUENTIAL = ");

    for (i=0; i<n; i++) {
        sum += v[i];
    }

    printf("%ld\n", sum);
    return sum;
}

/* Parallel but WRONG: concurrent updates of the accumulator. */
long sum_par1(const int *v, int n)
{
    long sum = 0;

    printf("*** [WRONG] PARALLEL WITH CONCURRENT UPDATES = ");

#pragma omp parallel
    {
        /* by default, `sum` is shared across threads */
        const int P = omp_get_num_threads();
        const int my_id = omp_get_thread_num();

        const int my_block_len = n/P;
        const int my_start = my_id * my_block_len;
        const int my_end = my_start + my_block_len;
        for (int my_i=my_start; my_i<my_end; my_i++) {
            sum += v[my_i];
        }
    }

    printf("%ld\n", sum);
    return sum;
}

/* Parallel with mutex on accumulator but WRONG partitioning if `P`
   does not divide `n`. */
long sum_par2(const int *v, int n)
{
    long sum = 0;

    printf("*** [WRONG] PARALLEL WITH MUTEX BUT WRONG PARTITIONING = ");

#pragma omp parallel
    {
        const int P = omp_get_num_threads();
        const int my_id = omp_get_thread_num();

        const int my_block_len = n/P;
        const int my_start = my_id * my_block_len;
        const int my_end = my_start + my_block_len;
        for (int my_i=my_start; my_i<my_end; my_i++) {
#pragma omp atomic
            sum += v[my_i];
        }
    }

    printf("%ld\n", sum);
    return sum;
}

/* Parallel with mutex on accumulator; not efficient, the mutex is the
   bottleneck. */
long sum_par3(const int* v, int n)
{
    long sum = 0;

    printf("*** PARALLEL WITH MUTEX ON ACCUMULATOR = ");

#pragma omp parallel
    {
        const int P = omp_get_num_threads();
        const int my_id = omp_get_thread_num();

        const int my_start = (n * my_id) / P;
        const int my_end = (n * (my_id+1)) / P;
        for (int my_i=my_start; my_i<my_end; my_i++) {
#pragma omp atomic
            sum += v[my_i];
        }
    }

    printf("%ld\n", sum);
    return sum;
}

/* Parallel with local accumulators and mutex on the global
   accumulator. This is more efficient than the version above, since
   it reduces the pressure on the mutex. */
long sum_par4(const int *v, int n)
{
    long sum = 0;

    printf("*** PARALLEL WITH LOCAL COUNTERS AND MUTEX ON GLOBAL COUNTER = ");

#pragma omp parallel
    {
        const int P = omp_get_num_threads();
        const int my_id = omp_get_thread_num();

        const int my_start = (n * my_id) / P;
        const int my_end = (n * (my_id+1)) / P;
        long my_sum = 0;
        for (int my_i=my_start; my_i<my_end; my_i++) {
            my_sum += v[my_i];
        }
#pragma omp atomic
        sum += my_sum;
    }

    printf("%ld\n", sum);
    return sum;
}

/* Parallel without mutex - WRONG without barrier synchronization. */
long sum_par5(const int *v, int n)
{
    const int num_threads = omp_get_max_threads();
    long psum[num_threads];
    long sum = 0;

    printf("*** [WRONG] PARALLEL WITH NO MUTEX BUT MISSING BARRIER SYNCHRONIZATION = ");

#pragma omp parallel
    {
        const int P = omp_get_num_threads();
        const int my_id = omp_get_thread_num();

        const int my_start = (n * my_id) / P;
        const int my_end = (n * (my_id+1)) / P;
        long my_sum = 0;
        for (int my_i=my_start; my_i<my_end; my_i++) {
            my_sum += v[my_i];
        }
        psum[my_id] = my_sum;
        if (my_id == 0)
            for (int my_i=0; my_i<P; my_i++)
                sum += psum[my_i];
    }

    printf("%ld\n", sum);
    return sum;
}

/* Parallel without mutex, with barrier synchronization. */
long sum_par6(const int *v, int n)
{
    const int P = omp_get_max_threads();
    long psum[P];
    long sum = 0;

    printf("*** PARALLEL WITH NO MUTEX, WITH BARRIER SYNCHRONIZATION = ");

#pragma omp parallel
    {
        const int my_id = omp_get_thread_num();

        const int my_start = (n * my_id) / P;
        const int my_end = (n * (my_id+1)) / P;
        long my_sum = 0;
        for (int my_i=my_start; my_i<my_end; my_i++) {
            my_sum += v[my_i];
        }
        psum[my_id] = my_sum;
        /* Note: in OpenMP there is an implicit barrier at the end of
           a parallel region; therefore, this code could be simplified
           by moving the following instructions outside the parallel
           region (the "if" statement becomes no longer necessary). */
#pragma omp barrier
        if (my_id == 0) {
            for (int my_i=0; my_i<P; my_i++)
                sum += psum[my_i];
        }
    }
    printf("%ld\n", sum);
    return sum;
}

/* Parallel with reduction on local sums. */
long sum_par7(const int *v, int n)
{
   long sum = 0;

   printf("*** PARALLEL REDUCTION OF LOCAL SUMS = ");

#pragma omp parallel reduction(+:sum)
    {
        const int P = omp_get_num_threads();
        const int my_id = omp_get_thread_num();

        const int my_start = (n * my_id) / P;
        const int my_end = (n * (my_id+1)) / P;
        long my_sum = 0;
        for (int my_i=my_start; my_i<my_end; my_i++) {
            my_sum += v[my_i];
        }
        sum += my_sum;
    }

    printf("%ld\n", sum);
    return sum;
}

/* Parallel with global reduction; this is how this function should be
   implemented in OpenMP. */
long sum_par8(const int *v, int n)
{
    long sum = 0;

    printf("*** PARALLEL REDUCTION = ");

#pragma omp parallel for reduction(+:sum)
    for (int i=0; i<n; i++) {
        sum += v[i];
    }

    printf("%ld\n", sum);
    return sum;
}

int main(int argc, char *argv[])
{
    int len;
    int *v;

    if (argc == 2)
        len = atoi(argv[1]);
    else
        len = 939391; /* a prime number */

    printf("\narray length = %d\n\n", len);

    v = (int*)malloc(len * sizeof(*v));
    assert(v != NULL);

    srand(123);
    for (int i=0; i<len; i++)
    	v[i] = rand()%2;

    sum_seq(v, len);
    sum_par1(v, len);
    sum_par2(v, len);
    sum_par3(v, len);
    sum_par4(v, len);
    sum_par5(v, len);
    sum_par6(v, len);
    sum_par7(v, len);
    sum_par8(v, len);

    free(v);

    return EXIT_SUCCESS;
}
