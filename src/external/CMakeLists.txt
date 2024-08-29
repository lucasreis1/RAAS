set(MKL_THREADING "sequential")
set(MKL_ARCH "intel64")
set(MKL_LINK "dynamic")
set(MKL_INTERFACE "lp64")

find_package(MKL QUIET PATHS /opt/intel/oneapi/mkl/latest)

add_library(raas::mkl INTERFACE IMPORTED)
#target_include_directories(raas::mkl INTERFACE ${MKL_INCLUDE_DIR})
target_include_directories(raas::mkl INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
#target_link_libraries(raas::mkl INTERFACE ${MKL_LIBRARIES})
target_link_libraries(raas::mkl INTERFACE $<LINK_ONLY:MKL::MKL>)
foreach(MKL_LIB IN LISTS MKL_LIBRARIES)
  if(EXISTS "${MKL_LIB}")
    get_filename_component(MKL_LINK_DIR "${MKL_LIB}" DIRECTORY)
    if(IS_DIRECTORY "${MKL_LINK_DIR}")
      target_link_directories(raas::mkl INTERFACE "${MKL_LINK_DIR}")
    endif()
  endif()
endforeach()

set_property(
  TARGET raas::mkl PROPERTY INTERFACE_LINK_DIRECTORIES
  ${MKL_ROOT}/lib ${MKL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/intel64_win ${MKL_ROOT}/lib/win-x64)

if(UNIX)
  if(USE_STATIC_MKL)
    foreach(MKL_LIB_PATH IN LISTS MKL_LIBRARIES)
      if(NOT EXISTS "${MKL_LIB_PATH}")
        continue()
      endif()

      get_filename_component(MKL_LIB_NAME "${MKL_LIB_PATH}" NAME)

      # Match archive libraries starting with "libmkl_"
      if(MKL_LIB_NAME MATCHES "^libmkl_" AND MKL_LIB_NAME MATCHES ".a$")
        target_link_options(raas::mkl INTERFACE "-Wl,--exclude-libs,${MKL_LIB_NAME}")
      endif()
    endforeach()
  endif()
endif()

add_library(common STATIC common.c)
target_link_libraries(common raas::mkl)

add_library(approx_gemm STATIC approx_gemm.cpp)
target_link_libraries(approx_gemm PRIVATE common)
