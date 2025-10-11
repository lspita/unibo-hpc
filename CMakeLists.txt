cmake_minimum_required(VERSION 3.24.0)

function(set_default var val)
    if(NOT ${var})
        set(${var} ${val} PARENT_SCOPE)
    endif()
endfunction()

set(SRC_DIR src)
set(INCLUDE_DIR include)
set(LIB_DIR lib)
set(EXEC_PATTERNS main)
set(TEST_PATTERNS *.test)
set(CMAKE_DIR cmake)
set(CMAKE_TEMPLATES_DIR templates)
set(TARGETS_DIR targets)
set(CLANG_TIDY clang-tidy)
set_default(CMAKE_BUILD_TYPE Debug)

include("${CMAKE_DIR}/setup.cmake")

function(get_arg index outvar default)
    list(LENGTH ARGN n)
    if (n GREATER ${index})
        list(GET ARGN ${index} val)
    else()
        set(val ${default})
    endif()
    set(${outvar} "${val}" PARENT_SCOPE)
endfunction()

function(abs_path outvar path)
    set(${outvar} "${CMAKE_SOURCE_DIR}/${path}" PARENT_SCOPE)
endfunction()

function(remove_regex outvar reg value)
    string(REGEX REPLACE "${reg}" "" res "${value}")
    set(${outvar} "${res}" PARENT_SCOPE)
endfunction()

function(src_path outvar path)
    remove_regex(res "^${CMAKE_SOURCE_DIR}\\/?" "${path}")
    remove_regex(res "^${SRC_DIR}\\/?" "${res}")
    set(${outvar} "${res}" PARENT_SCOPE)
endfunction()

function(cmake_template_path outvar path)
    remove_regex(res "^${CMAKE_SOURCE_DIR}\\/?" "${path}")
    remove_regex(res "^${CMAKE_DIR}\\/?" "${res}")
    remove_regex(res "^${CMAKE_TEMPLATES_DIR}\\/?" "${res}")
    set(${outvar} "${res}" PARENT_SCOPE)
endfunction()

function(check_exists path)
    if (NOT EXISTS "${path}")
        message(FATAL_ERROR "Path ${path} does not exist")
    endif()
endfunction()

function(check_isdir path)
    if (NOT IS_DIRECTORY "${path}")
        message(FATAL_ERROR "Path ${path} is not a directory")
    endif()
endfunction()

function(check_isnotdir path)
    if (IS_DIRECTORY "${path}")
        message(FATAL_ERROR "Path ${path} is not a directory")
    endif()
endfunction()

function(find_sources outvar dir)
    set(match_patterns)
    list(LENGTH ARGN args_len)
    if (args_len GREATER 0)
        set(patterns ${ARGN})
    else()
        set(patterns *)
    endif()
    foreach(ext IN LISTS SOURCES_EXTENSIONS)
        foreach(pattern IN LISTS patterns)
            list(APPEND match_patterns "${dir}/${pattern}.${ext}")
        endforeach()
    endforeach()
    file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${match_patterns})
    set(${outvar} "${sources}" PARENT_SCOPE)
endfunction()

# create project
project("${PROJECT_NAME}")
get_property(PROJECT_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

set(SOURCES_EXTENSIONS)
foreach(lang IN LISTS PROJECT_LANGUAGES)
    list(APPEND SOURCES_EXTENSIONS ${CMAKE_${lang}_SOURCE_FILE_EXTENSIONS})
endforeach()

set(EXEC_SOURCE_NAMES_LIST)
foreach(ext IN LISTS SOURCES_EXTENSIONS)
    list(APPEND EXEC_SOURCE_NAMES_LIST "${EXEC_SOURCE_NAME}.${ext}")
endforeach()

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# make dirs full-paths
abs_path(SRC_DIR_ABS ${SRC_DIR})
abs_path(INCLUDE_DIR_ABS ${INCLUDE_DIR})
abs_path(LIB_DIR_ABS ${LIB_DIR})
set(TARGETS_DIR_ABS "${CMAKE_BINARY_DIR}/${TARGETS_DIR}")
check_exists(${SRC_DIR_ABS})
check_isdir(${SRC_DIR_ABS})

# get source files
find_sources(MAIN_SOURCES ${SRC_DIR_ABS} ${EXEC_PATTERNS})
find_sources(SRC_SOURCES ${SRC_DIR_ABS})
find_sources(TEST_SOURCES ${SRC_DIR_ABS} ${TEST_PATTERNS})
list(REMOVE_ITEM SRC_SOURCES ${MAIN_SOURCES})
list(REMOVE_ITEM SRC_SOURCES ${TEST_SOURCES})
list(APPEND PROJECT_SOURCES ${SRC_SOURCES})

# create lib
set(PROJECT_LIB "${PROJECT_NAME}-lib")
set(LIB_DIR_EXISTS false)
if (EXISTS "${LIB_DIR_ABS}")
    check_isdir("${LIB_DIR_ABS}")
    set(LIB_DIR_EXISTS true)
    find_sources(LIB_SOURCES "${LIB_DIR_ABS}")
    list(APPEND PROJECT_SOURCES ${LIB_SOURCES})
endif()
list(LENGTH PROJECT_SOURCES cxx_sources_len)
if (cxx_sources_len GREATER 0)
    add_library("${PROJECT_LIB}" STATIC ${PROJECT_SOURCES})
    target_include_directories("${PROJECT_LIB}" PRIVATE "${SRC_DIR_ABS}")
else()
    add_library("${PROJECT_LIB}" INTERFACE)
endif()
if (EXISTS "${INCLUDE_DIR_ABS}")
    check_isdir("${INCLUDE_DIR_ABS}")
    target_include_directories("${PROJECT_LIB}" PUBLIC "${INCLUDE_DIR_ABS}")
endif()
if (${LIB_DIR_EXISTS})
    target_include_directories("${PROJECT_LIB}" PUBLIC "${LIB_DIR_ABS}")
endif()

function(make_executable outexe outid outpath filepath)
    get_filename_component(filename "${filepath}" NAME)
    get_filename_component(exename "${filepath}" NAME_WLE)
    get_filename_component(exedir "${filepath}" DIRECTORY)
    set(is_root_exe "${exename}" STREQUAL "${SRC_DIR}" AND "${exedir}" STREQUAL "${CMAKE_SOURCE_DIR}")
    if (NOT ${is_root_exe} AND NOT IS_DIRECTORY "${filepath}" AND "${filename}" IN_LIST EXEC_SOURCE_NAMES_LIST)
        get_filename_component(exename "${exedir}" NAME)
        get_filename_component(exedir "${exedir}" DIRECTORY)
    endif()
    src_path(exepath "${exedir}/${exename}")
    string(REPLACE "/" "_" exeid "${exepath}")

    add_executable("${exeid}" "${filepath}")
    get_arg(0 link_lib true ${ARGN})
    if (${link_lib})
        target_link_libraries(${exeid} PRIVATE ${PROJECT_LIB})
    endif()
    # https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#target-properties
    src_path(bindir "${exedir}")
    set(bindir "${TARGETS_DIR_ABS}/${bindir}")
    set_target_properties(${exeid} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${bindir}
        OUTPUT_NAME ${exename}
    )
    set(${outexe} ${${exeid}} PARENT_SCOPE) 
    set(${outid} ${exeid} PARENT_SCOPE)
    set(${outpath} ${exepath} PARENT_SCOPE) 
endfunction()

# create main executables
foreach(main_source IN LISTS MAIN_SOURCES)
    make_executable(mainexe mainid mainpath ${main_source})
endforeach()

# enable testing
include(CTest)
enable_testing()

# create test executables
foreach(test_file IN LISTS TEST_SOURCES)
    make_executable(testexe testid testpath ${test_file})
    add_test(NAME "${testpath}" COMMAND ${testid})
endforeach()

include("${CMAKE_DIR}/config.cmake" OPTIONAL)

# generate files from templates
set(CMAKE_TEMPLATES_PATH "${CMAKE_DIR}/${CMAKE_TEMPLATES_DIR}")
file(GLOB_RECURSE TEMPLATE_FILES CONFIGURE_DEPENDS ${CMAKE_TEMPLATES_PATH}/*)
foreach(template_file IN LISTS TEMPLATE_FILES)
    cmake_template_path(template_name ${template_file})
    abs_path(template_name_abs "${template_name}")
    configure_file(${template_file} ${template_name_abs} @ONLY)
endforeach()
