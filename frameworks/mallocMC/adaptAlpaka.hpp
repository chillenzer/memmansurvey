#pragma once

#include <alpaka/alpaka.hpp>
#include <tuple>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;

auto adaptAlpaka() {
  auto const platform = alpaka::Platform<Acc>{};
  auto const dev = alpaka::getDevByIdx(platform, 0);
  auto queue = alpaka::Queue<Acc, alpaka::NonBlocking>{dev};
  return std::make_tuple(dev, queue);
}
