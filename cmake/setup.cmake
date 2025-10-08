#[[
Required:
- PROJECT_NAME (string) project name

Variables:
- PROJECT_LANGUAGES (list, default=C CXX) project LANGUAGES property value
- SOURCES_EXTENSIONS (list, default=c cpp cxx cc) list of possible file extensions for source files
- EXEC_SOURCE_NAME (string, default=main) name of sources that become executables (without file extension) 
- SRC_DIR (string, default=src) sources directory
- INCLUDE_DIR (string, default=include) public include directory
- LIB_DIR (string, default=lib) external lib files
- TARGETS_DIR (string, default=targets) binary directory relative path containing all targets
- CMAKE_DIR (string, default=cmake) cmake config files directory
- CMAKE_TEMPLATES_DIR (string, default=cmake) CMAKE_DIR subdirectory containing template files
- TEST_EXTRA_EXT (string, default=test) extra file extension for tests (es. test -> mytest.test.c)
- CLANG_TIDY (string, default=clang-tidy) clang tidy executable, unset to stop using clang tidy
]]#

set(PROJECT_NAME cxx-project-template)
set(PROJECT_VERSION 0.1.0)
set(PROJECT_DESCRIPTION "C/C++ project template")
set(PROJECT_HOMEPAGE_URL https://github.com/lspita/cxx-project-template)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
