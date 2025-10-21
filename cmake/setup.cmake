#[[
Required:
- PROJECT_NAME (string) project name

Variables:
- SRC_DIR (string, default=src) sources directory
- INCLUDE_DIR (string, default=include) public include directory
- LIB_DIR (string, default=lib) external lib files
- EXEC_PATTERNS (list, default=main) list of patterns matching sources that are executables (without last file extension)
- TEST_PATTERNS (list, default=*.test) list of patterns matching sources that are tests (without last file extension)
- TARGETS_DIR (string, default=targets) binary directory relative path containing all targets
- CMAKE_DIR (string, default=cmake) cmake config files directory
- CMAKE_TEMPLATES_DIR (string, default=cmake) CMAKE_DIR subdirectory containing template files
- CLANG_TIDY (string, default=clang-tidy) clang tidy executable, unset to stop using clang tidy
]]#

set(PROJECT_NAME unibo-hpc)
set(PROJECT_VERSION 0.1.0)
set(PROJECT_DESCRIPTION "High-Performance Computing course @ Computer Science and Engineering, UniBo, Cesena Campus")
set(PROJECT_HOMEPAGE_URL https://github.com/lspita/unibo-hpc)
enable_language(C)
enable_language(CXX)
enable_language(CUDA)

foreach(lang C CXX)
    set(CMAKE_${lang}_HOST_COMPILER ${CMAKE_${lang}_COMPILER})
endforeach()

set(CMAKE_C_STANDARD 99)
