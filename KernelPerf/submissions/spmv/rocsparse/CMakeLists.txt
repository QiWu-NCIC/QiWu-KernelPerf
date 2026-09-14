cmake_minimum_required(VERSION 3.21)
project(qiwu_rocsparse_spmv_plugin LANGUAGES CXX HIP)

find_package(hip CONFIG REQUIRED)
find_path(ROCSPARSE_INCLUDE_DIR rocsparse/rocsparse.h REQUIRED)
find_library(ROCSPARSE_LIBRARY NAMES rocsparse REQUIRED)

set(QIWU_PLUGIN_ENTRY "variants/default.cu" CACHE STRING "Plugin entry source")
set(QIWU_PLUGIN_SOURCES "${QIWU_PLUGIN_ENTRY}")

function(qiwu_add_spmv_target dtype)
  add_library(qiwu_rocsparse_${dtype} STATIC ${QIWU_PLUGIN_SOURCES})
  set_source_files_properties(${QIWU_PLUGIN_SOURCES} PROPERTIES LANGUAGE HIP)
  target_compile_features(qiwu_rocsparse_${dtype} PUBLIC cxx_std_17)
  target_compile_definitions(qiwu_rocsparse_${dtype} PUBLIC QIWU_BACKEND_HIP)
  if(dtype STREQUAL "fp64")
    target_compile_definitions(qiwu_rocsparse_${dtype} PUBLIC QIWU_SPMV_FP64=1)
  endif()
  target_include_directories(qiwu_rocsparse_${dtype}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${ROCSPARSE_INCLUDE_DIR}"
  )
  target_link_libraries(qiwu_rocsparse_${dtype} PUBLIC hip::host "${ROCSPARSE_LIBRARY}")
  set_target_properties(qiwu_rocsparse_${dtype} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_executable(qiwu_rocsparse_example_${dtype} examples/standalone.cu)
  set_source_files_properties(examples/standalone.cu PROPERTIES LANGUAGE HIP)
  target_compile_definitions(qiwu_rocsparse_example_${dtype} PRIVATE QIWU_BACKEND_HIP)
  if(dtype STREQUAL "fp64")
    target_compile_definitions(qiwu_rocsparse_example_${dtype} PRIVATE QIWU_SPMV_FP64=1)
  endif()
  target_link_libraries(qiwu_rocsparse_example_${dtype} PRIVATE qiwu_rocsparse_${dtype})
endfunction()

qiwu_add_spmv_target(fp32)
qiwu_add_spmv_target(fp64)
