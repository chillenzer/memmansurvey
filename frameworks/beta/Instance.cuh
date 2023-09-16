#pragma once
#include "TestInstance.cuh"


#include <gallatin/allocators/gallatin.cuh>

using beta_type = gallatin::allocators::Gallatin<16ULL*1024*1024, 16, 4096>;


struct MemoryManagerBETA : public MemoryManagerBase
{
	explicit MemoryManagerBETA(size_t instantiation_size) : MemoryManagerBase(instantiation_size)
	{

		//fun hack - first time running the file, print info.
		//is extra needed? +20ULL*16ULL*1024*1024 - looks like no
		beta = beta_type::generate_on_device(instantiation_size, 42, !initialized);

		initialized = true;
		
	}

	MemoryManagerBETA(const MemoryManagerBETA& src) : beta{src.beta}, IAMACOPY{true} {}


	~MemoryManagerBETA(){

		if (!IAMACOPY){

			beta_type::free_on_device(beta);
			cudaDeviceSynchronize();

		}

	};

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return beta->malloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		beta->free(ptr);
	};

	static bool initialized;

	beta_type * beta;
	bool IAMACOPY{false};

};

bool MemoryManagerBETA::initialized = false;
