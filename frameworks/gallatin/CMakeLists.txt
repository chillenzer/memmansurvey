cmake_minimum_required(VERSION 3.16)
project(memorymanagerregeff LANGUAGES CXX CUDA VERSION 1.0)


add_subdirectory(gallatin)

set(BASE_LOCATION "../..")
include(${BASE_LOCATION}/BaseCMake.cmake)

# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_LOCATION}/include)
include(${BASE_PATH}/addCUB.cmake)

add_executable(memorymanagergal_example
 Instance.cuh
 main.cu
)

target_link_libraries(memorymanagergal_example gallatin)
