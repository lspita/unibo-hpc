#!/bin/bash

# This script test the performance of P/E-cores in modern CPUs.  It
# launches another script `demo-strong-scaling.sh` with various
# combinations of the `OMP_SCHEDULE` and `OMP_PLACES` variables, and
# produces several output files for subsequent analysis.

# First version 2025-10-03 Moreno Marzolla <https://www.moreno.marzolla.name/>
# Last modified on 2025-10-03

function run_test() {
    echo "OMP_SCHEDULE=$OMP_SCHEDULE OMP_PLACES=$OMP_PLACES"
    ./demo-strong-scaling.sh
}

OMP_SCHEDULE=static run_test | tee test-cores-1.out
OMP_SCHEDULE=static OMP_PLACES="0:8:2,16:8:1,1:8:2" | tee test-cores-2.out
OMP_SCHEDULE=static OMP_PLACES="0:8:2,1:8:2,16:8:1" | tee test-cores-3.out
OMP_SCHEDULE=dynamic run_test | tee test-cores-4.out

