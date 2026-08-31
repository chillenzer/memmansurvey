#pragma once

#include "TestInstance.cuh"
#include "mallocMC/Instance.cuh"

// ScatterAlloc is built against the same mallocMC branch as FlatterScatter
// (the chillenzer fork), using the Scatter creation policy.
using MemoryManagerScatterAlloc =
    MemoryManagerMallocMC<mallocMC::CreationPolicies::Scatter<>>;
