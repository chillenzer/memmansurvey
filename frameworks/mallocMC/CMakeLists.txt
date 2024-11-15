cmake_minimum_required(VERSION 3.16)
project(memorymanagermallocmc LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_LOCATION "../..")
include(${BASE_LOCATION}/BaseCMake.cmake)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/mallocMC" "${CMAKE_CURRENT_BINARY_DIR}/mallocMC")

alpaka_add_executable(mallocMC_main main.cu)

target_include_directories(mallocMC_main PRIVATE ${BASE_LOCATION}/include)
target_link_libraries(mallocMC_main PRIVATE mallocMC)
