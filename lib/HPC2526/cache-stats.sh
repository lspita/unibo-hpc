#!/bin/sh

## Use this script to analyze the cache performance of an application.
##
## Usage: ./cache-stats.sh executable_name [args]
##
## WARNINGS:
## --------
##
## - which counters are available is architecture-dependent;
##
## - the meaning of each counter is architecture-dependent.
##
## ISSUES:
## ------
##
## - On Ubuntu 24.04 the "perf" executable is not installed properly;
##   see
##   <https://bugs.launchpad.net/ubuntu/+source/linux-hwe-6.14/+bug/2117147>
##   However, "perf" is indeed present, e.g., in
##   `/usr/lib/linux-tools-6.8.0-78/perf` (your location may vary).
##   You can therefore execute it by specifying the full path.
##
## - On Ubuntu 24.04, the kernel prevents ordinary users to launch
##   "perf"; the workaround is to run "perf" as the superuser, or
##   create a file `/etc/sysctl.d/local.conf` with the following
##   content:
##
##   ```
##   kernel.perf_event_paranoid = -1
##   ```
##
## - On modern processors with P-cores and E-cores, the following
##   command produces duplicate entries:
##
## ```
## $ sudo /usr/lib/linux-tools-6.8.0-78/perf stat -B -e task-clock,cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses ./matmul-plain
## Starting plain matrix-matrix multiply (n=1024)... done
##
## r[0][0] = 264.159186
## 	elapsed time = 2.14 s
##
##
##  Performance counter stats for './matmul-plain':
##
##           2.170,04 msec task-clock                       #    0,999 CPUs utilized
##      <not counted>      cpu_atom/cycles/                                                        (0,00%)
##     10.958.400.611      cpu_core/cycles/                 #    5,050 GHz
##      <not counted>      cpu_atom/cache-references/                                              (0,00%)
##      1.079.247.332      cpu_core/cache-references/       #  497,340 M/sec
##      <not counted>      cpu_atom/cache-misses/                                                  (0,00%)
##          1.018.771      cpu_core/cache-misses/           #    0,09% of all cache refs
##      <not counted>      cpu_atom/L1-dcache-loads/                                               (0,00%)
##     15.115.102.796      cpu_core/L1-dcache-loads/        #    6,965 G/sec
##    <not supported>      cpu_atom/L1-dcache-load-misses/
##        907.097.166      cpu_core/L1-dcache-load-misses/  #    6,00% of all L1-dcache accesses
##      <not counted>      cpu_atom/L1-dcache-stores/                                              (0,00%)
##      2.190.490.601      cpu_core/L1-dcache-stores/       #    1,009 G/sec
##    <not supported>      cpu_atom/L1-dcache-store-misses/
##    <not supported>      cpu_core/L1-dcache-store-misses/
##
##        2,171867936 seconds time elapsed
##
##        2,162325000 seconds user
##        0,009001000 seconds sys
## ```
##
##   The "cpu_atom" entries refer to E-cores, while the "cpu_core"
##   entries refer to P-cores. In the message above, E-cores appears
##   to have not been used, so the counters are reported as "<not
##   counted>" (if you re-run the program they may have a value). To
##   pin the program to a specific core we can use the taskset
##   utility, e.g.:
##
##   ```
##   taskset -c 0 perf ... ./matmul-plain
##   ```
##
##   will run "perf ... ./matmul-plain" on cpu #0 which, in my case,
##   is a P-core (check with `lscpu --all --extended`). You can pin
##   the executable to any core/set of cores you want.
##
## Last modified by Moreno Marzolla on 2025-08-25

perf stat -B -e task-clock,cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses "$@"
