# Find the CUB library
find_package(CUB REQUIRED QUIET)
# If the library is not found, use the shipped version
if(NOT CUB_FOUND)
  message(STATUS "CUB not found, using shipped version. This might lead to incompatibilities.")
  include_directories(${BASE_PATH}/externals/cub)
else()
  include_directories(${CUB_INCLUDE_DIRS})
endif()
