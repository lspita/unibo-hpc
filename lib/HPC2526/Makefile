## This Makefile compiles the source programs discussed during the
## High Performance Computing class, 2025/2026, laurea degree in
## computer science and engineering, University of Bologna, Italy.
##
## To compile MPI targets you need the `mpicc` compiler wrapper.
## To compile CUDA targets you need the `nvcc` compiler.
## To compile OpenCL targets you need a working OpenCl installation.
##
## Last modified 2025-07-04
## Moreno Marzolla <https://www.moreno.marzolla.name/>
##
## `make`	generates all targets
## `make help`	shows the available targets

EXE_OMP:=$(basename $(filter-out omp-bug1.c omp-bug2.c, $(wildcard omp-*.c))) arraySum
EXE_MPI:=$(basename $(wildcard mpi-*.c))
EXE_SERIAL:=trap odd-even prefix-sum matmul-plain matmul-transpose game-of-life reduction vla-demo game-of-life-new matmul-test matmul_test.class
EXE_SIMD:=$(basename $(wildcard simd-*.c)) simd-vsum-vector.lst simd-vsum-auto.lst
EXE_CUDA:=$(basename $(wildcard *.cu))
EXE_OPENCL:=$(basename $(wildcard opencl-*.c))
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_SIMD) $(EXE_OPENCL) $(EXE_CUDA)
DISTFILES:=$(wildcard *.cu *.c *.h *.cl *.sh *.ods *.py *.java) Makefile README cat-1344.pgm mandelbrot-set-demo.ggb
DISTNAME:=HPC2526
CFLAGS+=-std=c99 -Wall -Wpedantic
MPICC?=mpicc
NVCC?=nvcc
MPICC?=mpicc
NVCFLAGS+=-Wno-deprecated-gpu-targets

all: mpi omp serial simd opencl cuda

help:
	@echo
	@echo "Available targets:"
	@echo
	@echo "     help    prints this message"
	@echo "   serial    compile all serial programs"
	@echo "      omp    compile all OpenMP programs"
	@echo "      mpi    compile all MPI programs"
	@echo "   opencl    compile all OpenLC programs"
	@echo "     cuda    compil3 all CUDA programs"
	@echo "     simd    compile all SIMD programs"
	@echo "      all    compile everything (default)"
	@echo "    clean    cleanup temporary files"
	@echo
	@echo "To install MPI on Debian/Ubuntu:"
	@echo
	@echo "    sudo apt install mpi-default-bin mpi-default-dev"
	@echo
	@echo "To install OpenCl on Debian/Ubuntu:"
	@echo
	@echo "    sudo apt install pocl-opencl-icd    # if you have no GPU"
	@echo "    sudo apt install beignet-opencl-icd # if you have Intel HD GPUs"
	@echo "    sudo apt install libclc-amdgcn      # if you have AMD GPPUs"
	@echo "    sudo apt install nvidia-opencl-icd  # if you have NVidia GPUs"
	@echo
	@echo "If you have an NVidia GPU, you can download the CUDA SDK here"
	@echo
	@echo "    https://developer.nvidia.com/cuda-downloads"
	@echo

dist: $(DISTNAME).zip

$(DISTNAME).zip: $(DISTFILES)
	rm -r -f $(DISTNAME)
	mkdir $(DISTNAME)
	cd $(DISTNAME)
	ln $(DISTFILES) $(DISTNAME)
	zip -r $(DISTNAME).zip $(DISTNAME)

gfx.o: gfx.c gfx.h

omp-mandelbrot: omp-mandelbrot.o gfx.o
omp-mandelbrot: LDLIBS+=-lX11
omp-mandelbrot: LDFLAGS+=-fopenmp

matmul-plain: LDLIBS+=-lrt
matmul-plain: matmul.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

matmul-transpose: CFLAGS+=-DTRANSPOSE
matmul-transpose: matmul.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

matmul-test: CFLAGS+=-fopenmp -mavx2 -mfma -O2 -ffast-math

matmul_test.class: matmul_test.java
	javac $<

$(EXE_OMP): CFLAGS+=-fopenmp
$(EXE_OMP): LDLIBS+=-lrt
omp: $(EXE_OMP)

$(EXE_MPI): CC=$(MPICC)
mpi: $(EXE_MPI)

mpi-vecsum mpi-vecsum2: LDLIBS+=-lm

serial: $(EXE_SERIAL)

$(EXE_SIMD): CFLAGS += -march=native
simd: $(EXE_SIMD)

simd-vsum-vector: CFLAGS+=-O2 -lm

simd-vsum-vector.s: simd-vsum-vector.c
	gcc -S -c -fverbose-asm -g -O2 -march=native -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized $< -o $@

simd-vsum-intrinsics: CFLAGS+=-O2 -lm

simd-vsum-auto: CFLAGS+=-lm -O2 -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized

simd-vsum-auto.s: simd-vsum-auto.c
	gcc -S -c -fverbose-asm -g -O2 -march=native -funsafe-math-optimizations -ftree-vectorize -fopt-info-vec-missed -fopt-info-vec-optimized $< -o $@

%.lst: %.s
	as -alhnd $< > $@

$(EXE_OPENCL): CFLAGS+=simpleCL.c
$(EXE_OPENCL): LDLIBS+=-lOpenCL
opencl: $(EXE_OPENCL)

opencl-rotate opencl-matmul: LDLIBS+=-lm

cuda: $(EXE_CUDA)

cuda-rotate cuda-matmul: NVCFLAGS+=-lm

demo-cuda-tput: NVCFLAGS+=-Xcompiler -fopenmp -O2 -lm

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

clean:
	\rm -f $(EXE) *.o *~ *.pbm *.s *.lst *.out *.class

distclean: clean
	\rm -r -f $(DISTNAME) $(DISTNAME).zip
