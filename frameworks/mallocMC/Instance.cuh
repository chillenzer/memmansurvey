#pragma once

#include "TestInstance.cuh"

#include "adaptAlpaka.hpp"
#include "mallocMC/mallocMC.hpp"
#include <array>

namespace MC = mallocMC;

using mallocMC::CreationPolicies::FlatterScatter;
constexpr uint32_t const blocksize = 128U * 1024U * 1024U;
constexpr uint32_t const pagesize = 128U * 1024U;
constexpr uint32_t const wasteFactor = 2U;

// This happens to also work for the original Scatter algorithm, so we only
// define one.
struct FlatterScatterHeapConfig : FlatterScatter<>::Properties::HeapConfig {
  static constexpr auto accessblocksize = blocksize;
  static constexpr auto pagesize = ::pagesize;
  // Only used by original Scatter (but it doesn't hurt FlatterScatter to keep):
  static constexpr auto regionsize = 16;
  static constexpr auto wastefactor = wasteFactor;
  static constexpr bool resetfreedpages = false;
};

struct ShrinkConfig {
  static constexpr auto dataAlignment = 16;
};

using ScatterAllocator = MC::Allocator<
    Acc, MC::CreationPolicies::FlatterScatter<FlatterScatterHeapConfig>,
    MC::DistributionPolicies::Noop, MC::OOMPolicies::ReturnNull,
    MC::ReservePoolPolicies::AlpakaBuf<Acc>,
    mallocMC::AlignmentPolicies::Shrink<ShrinkConfig>>;

static auto [dev, queue] = adaptAlpaka();

struct MemoryManagerMallocMC : public MemoryManagerBase {
  explicit MemoryManagerMallocMC(size_t instantiation_size)
      : MemoryManagerBase(instantiation_size),
        sa{new ScatterAllocator(dev, queue, instantiation_size)},
        sah{sa->getAllocatorHandle()} {}

  ~MemoryManagerMallocMC() {
    if (!IAMACOPY) {
      delete sa;
    }
  }

  MemoryManagerMallocMC(const MemoryManagerMallocMC &src)
      : sa{src.sa}, sah{src.sah}, IAMACOPY{true} {}

  virtual __device__ __forceinline__ void *malloc(size_t size) override {
    std::array<std::byte, sizeof(Acc)> fakeAccMemory{};
    return sah.malloc(*reinterpret_cast<Acc *>(fakeAccMemory.data()), size);
  }

  virtual __device__ __forceinline__ void free(void *ptr) override {
    std::array<std::byte, sizeof(Acc)> fakeAccMemory{};
    sah.free(*reinterpret_cast<Acc *>(fakeAccMemory.data()), ptr);
  }

  ScatterAllocator *sa;
  ScatterAllocator::AllocatorHandle sah;
  bool IAMACOPY{false}; // TODO: That is an ugly hack so we don't get a double
                        // free when making a copy for the device
};
