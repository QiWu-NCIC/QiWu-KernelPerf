cmake_minimum_required(VERSION 3.20)
project(qiwu_spmv_plugin LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(QIWU_PLUGIN_ENTRY "" CACHE STRING "Plugin entry source")
if(NOT QIWU_PLUGIN_ENTRY)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/plugin.json" QIWU_PLUGIN_MANIFEST)
  string(JSON QIWU_PLUGIN_ENTRY GET "${QIWU_PLUGIN_MANIFEST}" entry_source)
endif()
set(QIWU_PLUGIN_COMPILE_UNITS "" CACHE STRING "Additional source files")
set(QIWU_PLUGIN_EXTRA_INCLUDE_DIRS "" CACHE STRING "Additional dependency include directories")
set(QIWU_PLUGIN_EXTRA_LIBRARIES "" CACHE STRING "Additional dependency libraries")
set(QIWU_PLUGIN_EXTRA_LINK_OPTIONS "" CACHE STRING "Additional linker options")
set(QIWU_PLUGIN_SOURCES "${QIWU_PLUGIN_ENTRY}" ${QIWU_PLUGIN_COMPILE_UNITS})

set(GHOST_ROOT "" CACHE PATH "Pinned GHOST installation prefix")
find_path(QIWU_GHOST_INCLUDE_DIR ghost.h HINTS "${GHOST_ROOT}/include" REQUIRED)
find_library(QIWU_GHOST_LIBRARY ghost
  HINTS "${GHOST_ROOT}/lib/ghost" "${GHOST_ROOT}/lib" REQUIRED)
find_library(QIWU_BLAS_LIBRARY NAMES openblas blas)
list(APPEND QIWU_PLUGIN_EXTRA_INCLUDE_DIRS "${QIWU_GHOST_INCLUDE_DIR}")
list(APPEND QIWU_PLUGIN_EXTRA_LIBRARIES
  "${QIWU_GHOST_LIBRARY}" hwloc CUDA::cublas CUDA::curand)
if(QIWU_BLAS_LIBRARY)
  list(APPEND QIWU_PLUGIN_EXTRA_LIBRARIES "${QIWU_BLAS_LIBRARY}")
endif()
list(APPEND QIWU_PLUGIN_EXTRA_LINK_OPTIONS "-Wl,--allow-shlib-undefined")


function(qiwu_add_spmv_target dtype)
  add_library(qiwu_spmv_${dtype} STATIC ${QIWU_PLUGIN_SOURCES})
  target_compile_features(qiwu_spmv_${dtype} PUBLIC cxx_std_17)
  target_include_directories(qiwu_spmv_${dtype}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" ${QIWU_PLUGIN_EXTRA_INCLUDE_DIRS}
  )
  target_link_libraries(qiwu_spmv_${dtype}
    PUBLIC ${QIWU_PLUGIN_EXTRA_LIBRARIES} CUDA::cudart CUDA::cusparse
  )
  target_link_options(qiwu_spmv_${dtype} PUBLIC ${QIWU_PLUGIN_EXTRA_LINK_OPTIONS})
  set_target_properties(qiwu_spmv_${dtype} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  if(dtype STREQUAL "fp64")
    target_compile_definitions(qiwu_spmv_${dtype} PUBLIC QIWU_SPMV_FP64=1)
  endif()

  add_executable(qiwu_spmv_example_${dtype} examples/standalone.cu)
  target_link_libraries(qiwu_spmv_example_${dtype} PRIVATE qiwu_spmv_${dtype})
  set_target_properties(qiwu_spmv_example_${dtype} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

qiwu_add_spmv_target(fp32)
qiwu_add_spmv_target(fp64)
