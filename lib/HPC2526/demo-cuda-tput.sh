#!/bin/bash

# Last updated on 2024-11-13
# Moreno Marzolla <https://www.moreno.marzolla.name/>

PROG=./demo-cuda-tput   # executable name
PROB_SIZES="128 256 385 612 1024 1538 2048 2560 3072 3584 4096" # problem sizes
NREPS=5                 # number of replications

if [ ! -f "$PROG" ]; then
    echo
    echo "FATAL: $PROG not found"
    echo
    exit 1
fi

\rm -f demo-cuda-tput.txt
echo -e "N\tT_CUDA\t\tT_OpenMP"
for N in $PROB_SIZES; do
    ./demo-cuda-tput $N $NREPS | tee -a demo-cuda-tput.txt
done
