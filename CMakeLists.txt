cmake_minimum_required(VERSION 3.24)

project(RAAS)

if (NOT DEFINED LLVM_DIR)
  if (EXISTS /usr/local/lib/cmake/llvm)
    set(LLVM_DIR "/usr/local/lib/cmake/llvm")
  else()
    message(WARNING "LLVM_DIR is not set. If compilation fails, consider specifying it using -DLLVM_DIR=/path/to/llvm/lib/cmake")
  endif()
else()
  message(STATUS "LLVM_DIR=${LLVM_DIR}")
endif()

# llvm required packages
find_package(LLVM REQUIRED CONFIG)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})

list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(raas src/RAAS.cpp)
target_link_libraries(raas PRIVATE JIT)
target_compile_options(raas PRIVATE "-fno-rtti")

find_package(Python REQUIRED COMPONENTS Interpreter Development)
add_subdirectory(pybind11)
add_library(embedding SHARED src/embedding.cpp)
set_target_properties(embedding PROPERTIES PUBLIC_HEADER 
  "src/embedding.h;include/instrumentation.h")
target_link_libraries(embedding PUBLIC JIT)
add_executable(py_emb src/python_embedder.cpp)
target_link_libraries(py_emb PRIVATE Python::Python embedding)
pybind11_add_module(raas_support MODULE src/raas_support.cpp)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (${CMAKE_C_COMPILER} STREQUAL "clang")
  set_property(DIRECTORY APPEND PROPERTY COMPILE_OPTIONS -fdiagnostics-color)
endif()


# export compile commands by default
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

set (CMAKE_BUILD_TYPE Release)

add_subdirectory(src/jit)
add_subdirectory(src/external)

install(TARGETS embedding JIT instrumentFn
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
  ARCHIVE DESTINATION lib/static
)

install(TARGETS raas_support
  LIBRARY DESTINATION ${Python_SITELIB}
)

install(TARGETS raas py_emb
  RUNTIME DESTINATION bin
)

INSTALL(PROGRAMS prepare_library.sh DESTINATION bin)

#FUTURE NOTES:
# if after setting to Release, you are missing symbols, CMake probably sets visibility for symbols to default. 
# This means some symbols required for this program to work crash. Use:
#set_property (TARGET py_emb PROPERTY CXX_VISIBILITY_PRESET default)
