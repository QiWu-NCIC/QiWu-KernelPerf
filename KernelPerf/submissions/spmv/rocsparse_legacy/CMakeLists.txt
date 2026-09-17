cmake_minimum_required(VERSION 3.21)
project(qiwu_rocsparse_legacy_spmv_plugin LANGUAGES CXX HIP)

find_package(hip CONFIG REQUIRED)
find_path(ROCSPARSE_INCLUDE_DIR rocsparse/rocsparse.h REQUIRED)
find_library(ROCSPARSE_LIBRARY NAMES rocsparse REQUIRED)

function(qiwu_add_spmv_target dtype)
  add_library(qiwu_rocsparse_legacy_${dtype} STATIC adapter.cu)
  set_source_files_properties(adapter.cu PROPERTIES LANGUAGE HIP)
  target_compile_features(qiwu_rocsparse_legacy_${dtype} PUBLIC cxx_std_17)
  target_compile_definitions(qiwu_rocsparse_legacy_${dtype} PUBLIC QIWU_BACKEND_HIP)
  if(dtype STREQUAL "fp64")
    target_compile_definitions(qiwu_rocsparse_legacy_${dtype} PUBLIC QIWU_SPMV_FP64=1)
  endif()
  target_include_directories(qiwu_rocsparse_legacy_${dtype}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${ROCSPARSE_INCLUDE_DIR}"
  )
  target_link_libraries(qiwu_rocsparse_legacy_${dtype}
    PUBLIC hip::host "${ROCSPARSE_LIBRARY}"
  )
  set_target_properties(qiwu_rocsparse_legacy_${dtype}
    PROPERTIES POSITION_INDEPENDENT_CODE ON
  )

  add_executable(qiwu_rocsparse_legacy_example_${dtype} examples/standalone.cu)
  set_source_files_properties(examples/standalone.cu PROPERTIES LANGUAGE HIP)
  target_compile_definitions(qiwu_rocsparse_legacy_example_${dtype} PRIVATE QIWU_BACKEND_HIP)
  if(dtype STREQUAL "fp64")
    target_compile_definitions(qiwu_rocsparse_legacy_example_${dtype} PRIVATE QIWU_SPMV_FP64=1)
  endif()
  target_link_libraries(qiwu_rocsparse_legacy_example_${dtype}
    PRIVATE qiwu_rocsparse_legacy_${dtype}
  )
endfunction()

qiwu_add_spmv_target(fp32)
qiwu_add_spmv_target(fp64)
