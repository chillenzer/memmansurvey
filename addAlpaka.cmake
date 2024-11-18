option(alpaka_ACC_GPU_CUDA_ENABLE "" ON)
option(alpaka_ACC_GPU_CUDA_ONLY_MODE "" ON)
set(alpaka_CXX_STANDARD 20)
set(alpaka_BUILD_EXAMPLES OFF)
set(BUILD_TESTING OFF)

add_subdirectory(
  ${BASE_PATH}frameworks/mallocMC/mallocMC/alpaka
 "${CMAKE_CURRENT_BINARY_DIR}/alpaka" EXCLUDE_FROM_ALL)

