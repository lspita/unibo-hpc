#!/bin/bash

# This script executes the parallel `omp-matmul` program with an
# increasing number of threads p, from 1 up to the number of logical
# cores (or up to the value of the environment variable MAX_CORES, if
# defined). For each run, we use the same input size so that the
# execution times can be used to compute the speedup and the strong
# scaling efficiency. Each run is repeated `NREPS` times; the script
# prints all individual execution times on standard output.
#
#-----------------------------------------------------------------------
#
# NOTE: the problem size `PROB_SIZE` is the number of rows (or
# columns) of the input matrices. You may want to change the value
# according to the performance of the CPU under test. Ideally, the
# problem size should be large enough to get a reasonable execution
# time (at least a few seconds, ideally 10s or more) on all tests.

# Last updated on 2025-10-02
# Moreno Marzolla <https://www.moreno.marzolla.name/>

PROG=./omp-matmul       # name of the executable
PROB_SIZE=2048          # problem size; you may want to change this
MAX_CORES=${MAX_CORES:-`cat /proc/cpuinfo | grep processor | wc -l`} # number of (logical) cores
NREPS=5                 # number of replications.

if [ ! -f "$PROG" ]; then
    echo
    echo "$PROG not found"
    echo
    exit 1
fi

echo -e "p\tt1\tt2\tt3\tt4\tt5"

for p in `seq $MAX_CORES`; do
    echo -n -e "$p"
    for rep in `seq $NREPS`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $PROB_SIZE | grep "Execution time" | sed 's/Execution time //' )"
        echo -n -e "\t${EXEC_TIME}"
    done
    echo ""
done
