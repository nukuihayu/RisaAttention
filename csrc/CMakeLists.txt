cmake_minimum_required(VERSION 3.26)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  # Source deployments build for the GPU visible on the build host. Native
  # keeps direct CMake builds single-architecture and emits no PTX fallback.
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(RISAAttention_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit 13.0 REQUIRED)
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

set(ATTN_NVCC_FLAGS
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
  -O3
)

add_library(attn_cuda_kernels OBJECT
  kernels/quant_v_int8.cu
  kernels/quant_qk_int8.cu
  kernels/attention_launcher.cu
  kernels/retained_mass_selector.cu
)
target_compile_options(attn_cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${ATTN_NVCC_FLAGS}>
)
target_include_directories(attn_cuda_kernels PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  ${CUDAToolkit_INCLUDE_DIRS}
)

nanobind_add_module(_C NB_STATIC LTO bindings.cpp)
target_sources(_C PRIVATE $<TARGET_OBJECTS:attn_cuda_kernels>)
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
