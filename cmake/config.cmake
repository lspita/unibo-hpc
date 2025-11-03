#[[
Functions
- abs_path(outvar path) prefix path with absolute project directory
- src_path(outvar path) prefix path with absolute source directory
- remove_regex(outvar reg value) remove what matches `reg` from `value`
- check_exists(path) error if path doesn't exist
- check_isdir(path) error if path is not a directory
- check_isnotdir(path) error if path is a directory

Variables:
- PROJECT_LIB (library) library with all source files
- SRC_SOURCES (list) list of all sources in SRC_DIR
- EXEC_SOURCES (list) list of all sources that are executables
- TEST_SOURCES (list) list of all sources that are tests
- SRC_DIR_ABS (string) abs path of SRC_DIR
- INCLUDE_DIR_ABS (string) abs path of INCLUDE_DIR
- LIB_DIR_ABS (string) abs path of LIB_DIR

Every test is named after the relative path from SRC_DIR.
Targets are located under bin/<preset>/targets
]]#

find_package(OpenMP REQUIRED)
find_package(X11 REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCL REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_LIB} PUBLIC
    OpenMP::OpenMP_C
    X11::X11
    MPI::MPI_C
    OpenCL::OpenCL
    CUDA::cudart
)

set(HPC_ROOT ${LIB_DIR}/HPC2526)
set(HPC_AVX_FLAGS_SOURCES ${HPC_ROOT}/matmul-test.c)

if (MSVC)
    target_compile_options(${PROJECT_LIB} PRIVATE /W4)
    set_source_files_properties(${HPC_AVX_FLAGS_SOURCES} PROPERTIES 
        COMPILE_FLAGS "/arch:AVX2"
    )
else()
    target_compile_options(${PROJECT_LIB} PRIVATE -Wall -Wextra -Wpedantic)
    set_source_files_properties(${HPC_AVX_FLAGS_SOURCES} PROPERTIES 
        COMPILE_FLAGS "-mavx2 -mfma"
    )
endif()

set(PREVENT_COMPILE_SOURCES
    ${HPC_ROOT}/omp-bug1.c
    ${HPC_ROOT}/omp-bug2.c
    ${HPC_ROOT}/omp-bug3.c
)
foreach(source IN LISTS PREVENT_COMPILE_SOURCES)
    set_source_files_properties(${source} PROPERTIES HEADER_FILE_ONLY ON)
endforeach()

