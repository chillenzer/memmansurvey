option(alpaka_ACC_GPU_CUDA_ENABLE "" ON)
option(alpaka_ACC_GPU_CUDA_ONLY_MODE "" ON)
set(alpaka_CXX_STANDARD 20)
set(alpaka_BUILD_EXAMPLES OFF)
set(BUILD_TESTING OFF)

macro(create_mallocMC_executables targets sources base_path)
  # The mallocMC fork must already have been added by the caller, since it is
  # shared between the mallocMC and ScatterAlloc builds.
  list(APPEND creation_policies FlatterScatter Scatter)

  # generates targets for executables like `m_synth_test_f` and so on...
  foreach(target source IN ZIP_LISTS targets sources)
    foreach(creation_policy IN LISTS creation_policies)
      string(SUBSTRING ${creation_policy} 0 1 first_char)
      string(TOLOWER ${first_char} lower_char)
      set(target_name "m_${target}_${lower_char}")

      string(TOUPPER ${creation_policy} upper_policy)

      add_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR}/${source})
      target_link_libraries(${target_name} PRIVATE mallocMC::mallocMC)
      target_compile_definitions(${target_name} PUBLIC TEST_MALLOCMC TEST_${upper_policy})
      set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES OFF)
    endforeach()
  endforeach()
endmacro()
