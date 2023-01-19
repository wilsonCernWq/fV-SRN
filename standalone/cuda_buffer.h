//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once
#ifndef HELPER_CUDA_BUFFER_H
#define HELPER_CUDA_BUFFER_H

#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <vector>

// #define CUDA_BUFFER_VERBOSE_MEMORY_ALLOCS

// ------------------------------------------------------------------
// CUDA Buffer
// ------------------------------------------------------------------

/*! simple wrapper for creating, and managing a device-side CUDA buffer */
struct CUDABuffer {
  size_t sizeInBytes = 0;
  void* d_ptr = nullptr;
  bool owned_data = false;

public:
	CUDABuffer() {}

	CUDABuffer& operator=(CUDABuffer&& other) 
  {
		std::swap(sizeInBytes, other.sizeInBytes);
		std::swap(d_ptr, other.d_ptr);
		return *this;
	}

	CUDABuffer(CUDABuffer&& other) 
  {
		*this = std::move(other);
	}

	CUDABuffer(const CUDABuffer& other) = delete;

	// Frees memory again
	~CUDABuffer() 
  {
		try {
			free();
		} catch (std::runtime_error error) {
			// Don't need to report on memory-free problems when the driver is shutting down.
			if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
				fprintf(stderr, "Could not free memory: %s\n", error.what());
			}
		}
	}

  void set_external(CUDABuffer& other)
  {
    sizeInBytes = other.sizeInBytes;
    d_ptr = other.d_ptr;
    owned_data = false;
  }

  // access raw pointer
  inline const CUdeviceptr& d_pointer() const { return (const CUdeviceptr&)d_ptr; }

  // re-size buffer to given number of bytes
  void resize(size_t size)
  {
    if (size == sizeInBytes) return;
    free(); alloc(size);
  }

  // set memory value
  void memset(int value)
  {
    CUDA_CHECK(cudaMemset((void*)d_ptr, value, sizeInBytes));
  }

  // allocate to given number of bytes
  void alloc(size_t size)
  {
    assert(d_ptr == nullptr);
    this->sizeInBytes = size;
    
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes));

    util::total_n_bytes_allocated() += sizeInBytes;

#ifdef CUDA_BUFFER_VERBOSE_MEMORY_ALLOCS
		printf("[mem] CUDABuffer alloc %s\n", util::prettyBytes(sizeInBytes).c_str());
#endif

    owned_data = true;
  }

  // free allocated memory
  void free()
  {
    if (owned_data && d_ptr) {
      CUDA_CHECK(cudaFree(d_ptr));
      util::total_n_bytes_allocated() -= sizeInBytes;
#ifdef CUDA_BUFFER_VERBOSE_MEMORY_ALLOCS
      printf("[mem] CUDABuffer free %s\n", util::prettyBytes(sizeInBytes).c_str());
#endif
    }
    d_ptr = nullptr;
    sizeInBytes = 0;
  }

  template<typename T>
  void alloc_and_upload(const std::vector<T>& vt)
  {
    resize(vt.size() * sizeof(T));
    upload((const T*)vt.data(), vt.size());
  }

  template<typename T>
  void alloc_and_upload(const T* ptr, size_t size)
  {
    resize(size * sizeof(T));
    upload((const T*)ptr, size);
  }

  template<typename T>
  void upload(const T* t, size_t count)
  {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(cudaMemcpy(d_ptr, (void*)t, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  template<typename T>
  void download(T* t, size_t count)
  {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(cudaMemcpy((void*)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
};

#endif // HELPER_CUDA_BUFFER_H
