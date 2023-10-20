#pragma once
#include "TestInstance.cuh"


#include <gallatin/allocators/gallatin.cuh>

using gal_type = gallatin::allocators::Gallatin<16ULL*1024*1024, 16, 4096>;


struct MemoryManagerGal : public MemoryManagerBase
{
	explicit MemoryManagerGal(size_t instantiation_size) : MemoryManagerBase(instantiation_size)
	{

		//fun hack - first time running the file, print info.
		//is extra needed? +20ULL*16ULL*1024*1024 - looks like no
		gal = gal_type::generate_on_device(instantiation_size, 42, !initialized);

		initialized = true;
		
	}

	MemoryManagerGal(const MemoryManagerGal& src) : gal{src.gal}, IAMACOPY{true} {}


	~MemoryManagerGal(){

		if (!IAMACOPY){

			gal_type::free_on_device(gal);
			cudaDeviceSynchronize();

		}

	};

	virtual __device__ __forceinline__ void* malloc(size_t size) override
	{
		return gal->malloc(size);
	}

	virtual __device__ __forceinline__ void free(void* ptr) override
	{
		gal->free(ptr);
	};

	static bool initialized;

	gal_type * gal;
	bool IAMACOPY{false};

};

bool MemoryManagerGal::initialized = false;
