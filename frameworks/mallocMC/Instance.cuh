#pragma once

#include "TestInstance.cuh"
#include "mallocMC/mallocMC.cuh"

template <typename T_CreationPolicy =
              mallocMC::CreationPolicies::FlatterScatter<>>
struct MemoryManagerMallocMC : public MemoryManagerBase {
  explicit MemoryManagerMallocMC(size_t instantiation_size)
      : MemoryManagerBase(instantiation_size),
        hostInfrastructure{
            new mallocMC::CudaHostInfrastructure<T_CreationPolicy>{
                instantiation_size}},
        mm{*hostInfrastructure} {}

  ~MemoryManagerMallocMC() {
    if (!IAMACOPY) {
      delete hostInfrastructure;
    }
  }

  MemoryManagerMallocMC(const MemoryManagerMallocMC &src)
      : hostInfrastructure{src.hostInfrastructure}, mm{*hostInfrastructure},
        IAMACOPY{true} {}

  virtual __device__ __forceinline__ void *malloc(size_t size) override {
    return mm.malloc(size);
  }

  virtual __device__ __forceinline__ void free(void *ptr) override {
    mm.free(ptr);
  }

  mallocMC::CudaHostInfrastructure<T_CreationPolicy> *hostInfrastructure;
  mallocMC::CudaMemoryManager<T_CreationPolicy> mm;
  bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double
                        // free when making a copy for the device
};

using MemoryManagerMallocMC_FlatterScatter =
    MemoryManagerMallocMC<mallocMC::CreationPolicies::FlatterScatter<>>;
using MemoryManagerMallocMC_Scatter =
    MemoryManagerMallocMC<mallocMC::CreationPolicies::Scatter<>>;
