#pragma once

#include "TestInstance.cuh"
#include "mallocMC/mallocMC.cuh"

namespace MC = mallocMC;

template <typename T_CreationPolicy = MC::CreationPolicies::FlatterScatter<>>
struct MemoryManagerMallocMC : public MemoryManagerBase {
  explicit MemoryManagerMallocMC(size_t instantiation_size)
      : MemoryManagerBase(instantiation_size),
        hostInfrastructure{new MC::CudaHostInfrastructure<T_CreationPolicy>(
            instantiation_size)},
        handle{hostInfrastructure->getAllocatorHandle()} {}

  ~MemoryManagerMallocMC() {
    if (!IAMACOPY) {
      delete hostInfrastructure;
    }
  }

  MemoryManagerMallocMC(const MemoryManagerMallocMC &src)
      : hostInfrastructure{src.hostInfrastructure}, handle{src.handle},
        IAMACOPY{true} {}

  virtual __device__ __forceinline__ void *malloc(size_t size) override {
    return handle.malloc(size);
  }

  virtual __device__ __forceinline__ void free(void *ptr) override {
    handle.free(ptr);
  }

  MC::CudaHostInfrastructure<T_CreationPolicy> *hostInfrastructure;
  MC::CudaHostInfrastructure<T_CreationPolicy>::AllocatorHandle handle;
  bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double
                        // free when making a copy for the device
};
