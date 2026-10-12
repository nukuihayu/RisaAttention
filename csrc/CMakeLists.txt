cmake_minimum_required(VERSION 3.26)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  # Source deployments build for the GPU visible on the build host. Native
  # keeps direct CMake builds single-architecture and emits no PTX fallback.
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(RisaAttention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit 12.8 REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

message(STATUS "RISA Attention CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(RISA_CUDA_FLAGS
  -DNDEBUG
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_BFLOAT16_OPERATORS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__
  -U__CUDA_NO_BFLOAT162_OPERATORS__
  -U__CUDA_NO_BFLOAT162_CONVERSIONS__
  --expt-relaxed-constexpr
  --expt-extended-lambda
  --use_fast_math
  # Compress embedded cubins, particularly useful for multi-SM wheels (CUDA 12.8+).
  --compress-mode=size
  -O3
)

add_library(risa_cuda_kernels OBJECT
  kernels/quantize_v.cu
  kernels/quantize_qk.cu
  kernels/quantize_gather.cu
  kernels/lse_correction.cu
  kernels/key_mask_tiles.cu
  kernels/attention_dispatch.cu
  kernels/sparse_selection.cu
)
target_compile_options(risa_cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${RISA_CUDA_FLAGS}>
)
# Keep the zero-query fast path for SM120-only builds. Fat binaries select
# the measured SM120 launch at runtime, without changing other devices.
set(RISA_EFFECTIVE_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
if(RISA_EFFECTIVE_ARCHITECTURES STREQUAL "native")
  set(RISA_EFFECTIVE_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
endif()
foreach(RISA_ARCH IN LISTS RISA_EFFECTIVE_ARCHITECTURES)
  if(RISA_ARCH MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 LESS 80)
    message(FATAL_ERROR "RISA requires SM80 or newer; unsupported architecture: ${RISA_ARCH}")
  endif()
endforeach()
if(RISA_EFFECTIVE_ARCHITECTURES MATCHES "^120(-real)?$")
  set_source_files_properties(kernels/quantize_v.cu kernels/quantize_gather.cu PROPERTIES
    COMPILE_DEFINITIONS RISA_V_QUANT_SM120=1)
endif()
target_include_directories(risa_cuda_kernels PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  ${CUDAToolkit_INCLUDE_DIRS}
)

# SM120a instructions stay in a real cubin; generic and multi-SM kernels keep
# their original architecture targets and require no extra libcuda dependency.
if(RISA_EFFECTIVE_ARCHITECTURES MATCHES "(^|;)120(a)?(-real)?(;|$)")
  add_library(risa_sm120_tma OBJECT kernels/dense_tma_sm120.cu)
  set_target_properties(risa_sm120_tma PROPERTIES CUDA_ARCHITECTURES "120a-real")
  target_compile_options(risa_sm120_tma PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${RISA_CUDA_FLAGS}>)
  target_include_directories(risa_sm120_tma PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(risa_cuda_kernels PRIVATE RISA_HAVE_SM120_TMA=1)
  if(RISA_EFFECTIVE_ARCHITECTURES MATCHES "^120(a)?(-real)?$")
    target_compile_definitions(risa_cuda_kernels PRIVATE RISA_SM120_ONLY=1)
    target_compile_definitions(risa_sm120_tma PRIVATE RISA_SM120_ONLY=1)
  endif()
endif()

# Keep packaging builds within the configured single-process memory budget.
nanobind_add_module(_C NB_STATIC bindings.cpp)
target_sources(_C PRIVATE $<TARGET_OBJECTS:risa_cuda_kernels>)
if(TARGET risa_sm120_tma)
  target_sources(_C PRIVATE $<TARGET_OBJECTS:risa_sm120_tma>)
endif()
target_compile_definitions(_C PRIVATE NB_STATIC)
target_include_directories(_C PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(_C PRIVATE CUDA::cudart_static)

if(MSVC)
  target_compile_options(_C PRIVATE /O2 /Zc:preprocessor)
  target_link_libraries(_C PRIVATE ws2_32)
else()
  target_compile_options(_C PRIVATE -O3 -fvisibility=hidden)
endif()

install(TARGETS _C LIBRARY DESTINATION risa_attention RUNTIME DESTINATION risa_attention)
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  configure_file(version.py.in "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py" DESTINATION risa_attention)
endif()
