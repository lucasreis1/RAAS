cmake_minimum_required(VERSION 3.10)

project(RAAS)

# llvm required packages
set(LLVM_DIR /mnt/linux/doutorado/llvm-installs/llvm-fork/build/lib/cmake/llvm)
find_package(LLVM REQUIRED CONFIG)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})
#add_library(embedding STATIC src/embedding.cpp)

add_executable(raas src/RAAS.cpp)
target_link_libraries(raas PRIVATE JIT)
target_compile_options(raas PRIVATE "-fno-rtti")

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
add_executable(py_emb src/python_embedder.cpp src/embedding.cpp)
target_link_libraries(py_emb PRIVATE Python::Python JIT)
pybind11_add_module(raas_support MODULE src/raas_support.cpp)

#target_link_options(py_emb PRIVATE "-Wl,-export-dynamic")
#set_property (TARGET py_emb PROPERTY CXX_VISIBILITY_PRESET default)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED 17)

set (CMAKE_CXX_COMPILER clang++)
set (CMAKE_C_COMPILER  clang)

set_property(DIRECTORY APPEND PROPERTY COMPILE_OPTIONS -fdiagnostics-color)

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# TODO: set to release for real tests
set (CMAKE_BUILD_TYPE Debug)

add_subdirectory(src/jit)
add_subdirectory(src/external)

#FUTURE NOTES:
# if after setting to Release, you are missing symbols, CMake probably sets visibility for symbols to default. 
# This means some symbols required for this program to work crash. Use:
#set_property (TARGET py_emb PROPERTY CXX_VISIBILITY_PRESET default)
