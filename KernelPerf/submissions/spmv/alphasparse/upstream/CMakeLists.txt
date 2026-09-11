cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project("AlphaSparse-Library" LANGUAGES CXX C)

# --- Basic compilation settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gdwarf-4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -gdwarf-4")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Build option definitions ---
option(ALPHA_BUILD_CUDA "Compile kernels for NVIDIA GPUs" OFF)
option(ALPHA_BUILD_HIP "Compile kernels for AMD or NVIDIA GPUs" ON)
option(ALPHA_BUILD_HYGON "Compile kernels for Hygon CPUs" OFF)
option(ALPHA_BUILD_ARM "Compile kernels for ARM CPUs" OFF)
option(ALPHA_BUILD_PLAIN "Compile serial kernels for x86 CPUs (Reference Implementation)" OFF)


# ====================================================================
# [Core change] Make the MKL dependency conditional
# Only check and pull in the MKL dependency when building the PLAIN or HYGON version
# ====================================================================
set(MKL_IS_REQUIRED OFF)
if(ALPHA_BUILD_PLAIN OR ALPHA_BUILD_HYGON)
    set(MKL_IS_REQUIRED ON)
endif()

if(MKL_IS_REQUIRED)
    message(STATUS "MKL support is required for the current build configuration (PLAIN or HYGON).")
    # Check the MKLROOT environment variable
    if(NOT DEFINED ENV{MKLROOT})
        message(FATAL_ERROR "MKLROOT environment variable is not set, but is required for PLAIN/HYGON builds. Please source the oneAPI setvars.sh script before running cmake.")
    endif()

    message(STATUS "Using MKL from MKLROOT: $ENV{MKLROOT}")

    # Add the MKL header and library paths
    include_directories($ENV{MKLROOT}/include)
    link_directories($ENV{MKLROOT}/lib/intel64)

    # Add the __MKL__ macro definition uniformly
    add_definitions(-D__MKL__)
else()
    message(STATUS "MKL support is NOT required for the current build configuration.")
endif()
# ======================= End of changes ============================


# --- Per-backend build logic ---

if(ALPHA_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set (CUDA_ARCH 70)
    add_subdirectory(cuda)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70")
    string(APPEND CMAKE_CUDA_FLAGS "  -Xcompiler -Wall -fPIC -D_FORCE_INLINES --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")
    string(APPEND CMAKE_CUDA_FLAGS " -I/home/guochengxin/cusplibrary-cuda10")
endif()

if(ALPHA_BUILD_HIP)
    set(HIP_CLANG_ROOT /opt/rocm) #change if needed
    set(CMAKE_HIP_HOST_COMPILER ${HIP_CLANG_ROOT}/llvm/bin/clang++)
    set(CMAKE_HIP_COMPILER ${HIP_CLANG_ROOT}/llvm/bin/clang++)
    set(HIP_CLANG_INCLUDE_PATH  ${HIP_CLANG_ROOT}/llvm/lib/clang/15.0.0/lib)
    enable_language(HIP)
    set (HIP_ARCH 70)    
    find_package(hip REQUIRED)
    find_package(hipsparse REQUIRED)
    find_package(rocprim REQUIRED)
    find_package(rccl REQUIRED)
    find_package(rocsparse REQUIRED)
    add_subdirectory(hip)
endif()

if(ALPHA_BUILD_DCU)
    enable_language(HIP)
    find_package(HIP REQUIRED)
    include_directories(${HIP_INCLUDE_DIRS})
    set (HIP_ARCH 70)
    set(CMAKE_CXX_COMPILER hipcc)
    set(__HIP_PLATFORM_AMD__ 1)
    find_package(hipsparse REQUIRED)
    find_package(rocprim REQUIRED)
    find_package(rccl REQUIRED)
    find_package(rocsparse REQUIRED)
    add_subdirectory(dcu)
endif()

if(ALPHA_BUILD_HYGON)
    # [Modified] -D__MKL__ no longer needs to be added here; it was moved to the conditional block at the top
    enable_language(C ASM)
    find_package(OpenMP REQUIRED)
    Set(CMAKE_C_FLAGS "$CMAKE_C_FLAGS} -fopenmp")
    Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    Set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_subdirectory(hygon)
endif()

if(ALPHA_BUILD_ARM)
    enable_language(C ASM)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_subdirectory(arm)
endif()

if(ALPHA_BUILD_PLAIN)
    # [Modified] -D__MKL__ no longer needs to be added here; it was moved to the conditional block at the top
    add_definitions(-D__PLAIN__)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_subdirectory(plain)
endif()


# --- Target library build and link ---
set(OBJECT_LIBS "")

if(ALPHA_BUILD_CUDA)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_cuda>)
endif()

if(ALPHA_BUILD_HYGON)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_hygon>)
endif()

if(ALPHA_BUILD_ARM)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_arm>)
endif()

if(ALPHA_BUILD_PLAIN)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_ref>)
endif()

if(ALPHA_BUILD_HIP)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_hip>)
endif()

if(OBJECT_LIBS)
    add_library(alphasparse SHARED ${OBJECT_LIBS})
    if(ALPHA_BUILD_HIP)
        target_link_libraries(alphasparse PUBLIC
            hip::host
            roc::hipsparse
            roc::rocprim
            roc::rccl
        )
    endif()
    target_include_directories(alphasparse PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    )
else()
    message(WARNING "No source directories are selected for building. No library will be created.")
endif()