cmake_minimum_required(VERSION 3.16)
project(memorymanagerscatteralloc LANGUAGES CXX CUDA VERSION 1.0)

set(BASE_LOCATION "../..")
include(${BASE_LOCATION}/BaseCMake.cmake)

# ScatterAlloc is built against the same chillenzer mallocMC fork as the main
# benchmark (with the Scatter creation policy).
include(${BASE_LOCATION}/addAlpaka.cmake)
add_subdirectory(${BASE_LOCATION}/frameworks/mallocMC/mallocMC ./mallocMC)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${BASE_LOCATION}/include)
include_directories(${BASE_LOCATION}/frameworks)

add_executable(memorymanagerscatteralloc
	main.cu
)
target_link_libraries(memorymanagerscatteralloc PRIVATE mallocMC::mallocMC)
