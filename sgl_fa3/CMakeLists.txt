# sgl-fa3: the FlashAttention-3 forward kernels SGLang serves with, packaged as a
# standalone wheel. The kernels are compiled from ../hopper; this directory
# holds the torch op registration, the Python wrapper, the tests and build.sh.
#
# The compile configuration mirrors what sglang-kernel built as `flash_ops`
# before FA3 moved out of it, so the kernels are the same: forward only, varlen
# only, sm_90a plus (optionally) sm_80/sm_86, no SM100+ targets.

cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(sgl-fa3 LANGUAGES CXX CUDA)

include(FetchContent)

# FetchContent_Populate(<name>) without a source is deprecated from 3.30.
cmake_policy(SET CMP0169 OLD)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

set(GITHUB_ARTIFACTORY "github.com" CACHE STRING "GitHub mirror URL")

# Python: the extension is built against the limited API (abi3).
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# CXX: Torch 2.14 headers reject anything below C++20 with a hard #error.
set(SGL_FA3_CXX_STANDARD "20" CACHE STRING "C++ standard used for host and CUDA compilation")
set(CMAKE_CXX_STANDARD ${SGL_FA3_CXX_STANDARD})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "Detected CUDAToolkit_VERSION=${CUDAToolkit_VERSION}")
if (CUDAToolkit_VERSION VERSION_LESS "12.4")
    message(FATAL_ERROR "sgl-fa3 needs CUDA >= 12.4 for sm_90a, got ${CUDAToolkit_VERSION}")
endif()

# Torch
find_package(Torch 2.11 REQUIRED)

# Torch's CMake config appends `-gencode` flags for the local GPU; drop them so
# only the targets listed below are compiled.
string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (TORCH_CXX11_ABI STREQUAL "0")
    message(STATUS "Using old C++ ABI (-D_GLIBCXX_USE_CXX11_ABI=0)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
    message(STATUS "Using new C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI=1)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
endif()

# CUTLASS: pinned to the commit sglang-kernel compiled FA3 against, not the
# csrc/cutlass submodule, so the generated code does not change with the move.
FetchContent_Declare(
    repo-cutlass
    URL      https://${GITHUB_ARTIFACTORY}/NVIDIA/cutlass/archive/57e3cfb47a2d9e0d46eb6335c3dc411498efa198.tar.gz
    URL_HASH SHA256=09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5
)
FetchContent_Populate(repo-cutlass)

set(SGL_FA3_HOPPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hopper")
if (NOT EXISTS "${SGL_FA3_HOPPER_DIR}/flash_api.cpp")
    message(FATAL_ERROR "FA3 sources not found at ${SGL_FA3_HOPPER_DIR}; build sgl_fa3 from a full checkout")
endif()

# ============================== Options ============================== #
option(SGL_FA3_ENABLE_SM8X         "Also compile the sm_80/sm_86 kernels"  ON)
option(SGL_FA3_ENABLE_SPARSE_MASK  "Compile the sparse mask kernels"       OFF)

# ============================== Flags ================================ #
set(SGL_FA3_CUDA_FLAGS
    "-DNDEBUG"
    "-O3"
    "-Xcompiler"
    "-fPIC"
    "-gencode=arch=compute_90a,code=sm_90a"
    "-std=c++${SGL_FA3_CXX_STANDARD}"
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTLASS_TEST_LEVEL=0"
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
    "-DCUTLASS_ENABLE_GDC_FOR_SM90"  # For PDL
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED"  # Necessary for the WGMMA shapes that we use
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--use_fast_math"
    "-Xcompiler=-Wconversion"
    "-Xcompiler=-fno-strict-aliasing"
)

# ============================== Sources ============================== #
set(FA3_INSTANTIATIONS "${SGL_FA3_HOPPER_DIR}/instantiations")

if (SGL_FA3_ENABLE_SM8X)
    list(APPEND SGL_FA3_CUDA_FLAGS
        "-gencode=arch=compute_80,code=sm_80"
        "-gencode=arch=compute_86,code=sm_86"
    )
    file(GLOB FA3_SM8X_GEN_SRCS "${FA3_INSTANTIATIONS}/flash_fwd_hdim*_sm80.cu")
    # Each *_softcapall_sm80.cu only #includes the per-softcap files that this glob
    # already selects, so compiling it as well would build those kernels twice.
    list(FILTER FA3_SM8X_GEN_SRCS EXCLUDE REGEX "_softcapall_sm80\\.cu$")
endif()

# The flash_fwd_hdimdiff_*_sm90.cu files only #include the flash_fwd_hdim{64_256,64_512,192_128}_*
# files, which the flash_fwd_hdim[0-9]* globs below already select. Do not add them, or those
# kernels are compiled and linked twice.
file(GLOB FA3_BF16_GEN_SRCS
    "${FA3_INSTANTIATIONS}/flash_fwd_hdim[0-9]*_bf16*_sm90.cu")

# FP16: individual hdim files instead of hdimall, which crashes ptxas.
file(GLOB FA3_FP16_GEN_SRCS
    "${FA3_INSTANTIATIONS}/flash_fwd_hdim[0-9]*_fp16*_sm90.cu")

file(GLOB FA3_FP8_GEN_SRCS
    "${FA3_INSTANTIATIONS}/flash_fwd_hdim[0-9]*_e4m3*_sm90.cu")

# Q/K 64 with V 256 has no SGLang caller (absorbed MLA is V 512); FLASHATTENTION_DISABLE_HDIMDIFF64_256 below.
list(FILTER FA3_BF16_GEN_SRCS EXCLUDE REGEX "flash_fwd_hdim64_256_")
list(FILTER FA3_FP16_GEN_SRCS EXCLUDE REGEX "flash_fwd_hdim64_256_")

set(SGL_FA3_SOURCES
    "csrc/fa3_extension.cc"
    "${SGL_FA3_HOPPER_DIR}/flash_prepare_scheduler.cu"
    "${SGL_FA3_HOPPER_DIR}/flash_api.cpp"
    "${SGL_FA3_HOPPER_DIR}/flash_fwd_combine.cu"
    ${FA3_BF16_GEN_SRCS}
    ${FA3_FP16_GEN_SRCS}
    ${FA3_FP8_GEN_SRCS}
    ${FA3_SM8X_GEN_SRCS}
)

# ============================== Target =============================== #
Python_add_library(_C MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SGL_FA3_SOURCES})

target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SGL_FA3_CUDA_FLAGS}>)
target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/include
    ${repo-cutlass_SOURCE_DIR}/tools/util/include
    ${SGL_FA3_HOPPER_DIR}
)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} c10 cuda)

# Variants SGLang never reaches are compiled out; calls that would need one fail with a
# TORCH_CHECK in mha_fwd instead of silently running something else.
set(SGL_FA3_COMPILE_DEFS
    TORCH_TARGET_VERSION=0x020B000000000000ULL
    USE_CUDA
    FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_UNEVEN_K
    # No caller passes new K/V (or rotary) to flash_attn_with_kvcache; SGLang writes the KV cache itself.
    FLASHATTENTION_DISABLE_APPENDKV
    # Only multimodal_gen calls FA3 without cu_seqlens/seqused (for the static persistent
    # scheduler and 2-CTA clusters), always unsplit, non-paged and without softcap.
    FLASHATTENTION_VARLEN_ONLY_SPLIT_PAGED_SOFTCAP
    FLASHATTENTION_DISABLE_HDIMDIFF64_256
)
if (NOT SGL_FA3_ENABLE_SM8X)
    list(APPEND SGL_FA3_COMPILE_DEFS FLASHATTENTION_DISABLE_SM8x)
endif()
if (NOT SGL_FA3_ENABLE_SPARSE_MASK)
    list(APPEND SGL_FA3_COMPILE_DEFS FLASHATTENTION_DISABLE_SPARSE_MASK)
endif()
target_compile_definitions(_C PRIVATE ${SGL_FA3_COMPILE_DEFS})

install(TARGETS _C LIBRARY DESTINATION sgl_fa3)
