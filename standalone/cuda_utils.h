//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#pragma once

#include <stdexcept>
#include <atomic>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif 
#ifdef CUDA_CHECK_NOEXCEPT
#undef CUDA_CHECK_NOEXCEPT
#endif 
#ifdef CUDA_SYNC_CHECK
#undef CUDA_SYNC_CHECK
#endif 
#ifdef CUDA_SYNC_CHECK_NOEXCEPT
#undef CUDA_SYNC_CHECK_NOEXCEPT
#endif 
#ifdef OPTIX_CHECK
#undef OPTIX_CHECK
#endif 
#ifdef OPTIX_CHECK_NOEXCEPT
#undef OPTIX_CHECK_NOEXCEPT
#endif 

#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t error = call;                                                                  \
    if (error != cudaSuccess) {                                                                \
      const char* msg = cudaGetErrorString(error);                                             \
      fprintf(stderr, "CUDA error (%s: line %d): %s\n", __FILE__, __LINE__, msg);              \
      throw std::runtime_error(std::string("CUDA error: " #call " failed with error ") + msg); \
    }                                                                                          \
  } while (0)

#define CUDA_CHECK_NOEXCEPT(call) \
  do {                            \
    cudaError_t error = call;                                                                  \
    if (error != cudaSuccess) {                                                                \
      const char* msg = cudaGetErrorString(error);                                             \
      fprintf(stderr, "CUDA error (%s: line %d): %s\n", __FILE__, __LINE__, msg);              \
    }                                                                                          \
  } while (0)

#define CUDA_SYNC_CHECK()                                                                             \
  do {                                                                                                \
    cudaDeviceSynchronize();                                                                          \
    cudaError_t error = cudaGetLastError();                                                           \
    if (error != cudaSuccess) {                                                                       \
      const char* msg = cudaGetErrorString(error);                                                    \
      fprintf(stderr, "CUDA sync error (%s: line %d): %s\n", __FILE__, __LINE__, msg);                \
      throw std::runtime_error(std::string("CUDA cudaDeviceSynchronize() failed with error ") + msg); \
    }                                                                                                 \
  } while (0)

#define CUDA_SYNC_CHECK_NOEXCEPT() \
  do {                             \
    cudaDeviceSynchronize();                                                                          \
    cudaError_t error = cudaGetLastError();                                                           \
    if (error != cudaSuccess) {                                                                       \
      const char* msg = cudaGetErrorString(error);                                                    \
      fprintf(stderr, "CUDA sync error (%s: line %d): %s\n", __FILE__, __LINE__, msg);                \
    }                                                                                                 \
  } while (0)

#define OPTIX_CHECK(call)                                                                      \
  do {                                                                                         \
    OptixResult res = call;                                                                    \
    if (res != OPTIX_SUCCESS) {                                                                \
      fprintf(stderr, "OptiX call (%s) failed with %d (line %d)\n", #call, res, __LINE__);     \
      throw std::runtime_error(std::string("OptiX call (") + #call + std::string(") failed")); \
    }                                                                                          \
  } while (0)

#define OPTIX_CHECK_NOEXCEPT(call) \
  do {                             \
    OptixResult res = call;        \
  } while (0)


#ifdef __NVCC__
#define CUDA_UTIL_BOTH_INLINE __forceinline__ __device__ __host__
#else
#define CUDA_UTIL_BOTH_INLINE 
#endif

namespace util {

template <typename T>
CUDA_UTIL_BOTH_INLINE T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
CUDA_UTIL_BOTH_INLINE T next_multiple(T val, T divisor) {
	return div_round_up(val, divisor) * divisor;
}

constexpr uint32_t n_threads_linear = 128;
constexpr uint32_t n_threads_bilinear = 16;
constexpr uint32_t n_threads_trilinear = 8;

#ifdef __NVCC__

//. ======================================================================== //
// linear version 
//. ======================================================================== //

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
	return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
	if (n_elements <= 0) {
		return;
	}
	kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>(n_elements, args...);
}

//. ======================================================================== //
// bilinear version 
//. ======================================================================== //

template<typename T>
constexpr uint32_t
n_blocks_bilinear(T n_elements)
{
  return ((uint32_t)n_elements + n_threads_bilinear - 1) / n_threads_bilinear;
}

template<typename K, typename T, typename... Types>
inline void
bilinear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T width, T height, Types... args)
{
  if (width <= 0 || height <= 0) {
    return;
  }
  dim3 block_size(n_threads_bilinear, n_threads_bilinear, 1);
  dim3 grid_size(n_blocks_bilinear(width), n_blocks_bilinear(height), 1);
  kernel<<<grid_size, block_size, shmem_size, stream>>>((uint32_t)width, (uint32_t)height, args...);
}

//. ======================================================================== //
// trilinear version 
//. ======================================================================== //

template<typename T>
constexpr uint32_t
n_blocks_trilinear(T n_elements)
{
  return ((uint32_t)n_elements + n_threads_trilinear - 1) / n_threads_trilinear;
}

template<typename K, typename T, typename... Types>
inline void
trilinear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T width, T height, T depth, Types... args)
{
  if (width <= 0 || height <= 0) {
    return;
  }
  dim3 block_size(n_threads_trilinear, n_threads_trilinear, n_threads_trilinear);
  dim3 grid_size(n_blocks_trilinear(width), n_blocks_trilinear(height), n_blocks_trilinear(depth));
  kernel<<<grid_size, block_size, shmem_size, stream>>>((uint32_t)width, (uint32_t)height, (uint32_t)depth, args...);
}

//. ======================================================================== //
//
//. ======================================================================== //

template <typename F>
__global__ void parallel_for_kernel(const size_t n_elements, F fun) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	fun(i);
}

template <typename F>
inline void parallel_for_gpu(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, F&& fun) {
	if (n_elements <= 0) {
		return;
	}
	parallel_for_kernel<F><<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>(n_elements, fun);
}

template <typename F>
inline void parallel_for_gpu(cudaStream_t stream, size_t n_elements, F&& fun) {
	parallel_for_gpu(0, stream, n_elements, std::forward<F>(fun));
}

template <typename F>
inline void parallel_for_gpu(size_t n_elements, F&& fun) {
	parallel_for_gpu(nullptr, n_elements, std::forward<F>(fun));
}

template <typename F>
__global__ void parallel_for_aos_kernel(const size_t n_elements, const uint32_t n_dims, F fun) {
	const size_t dim = threadIdx.x;
	const size_t elem = threadIdx.y + blockIdx.x * blockDim.y;
	if (dim >= n_dims) return;
	if (elem >= n_elements) return;

	fun(elem, dim);
}

template <typename F>
inline void parallel_for_gpu_aos(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
	if (n_elements <= 0 || n_dims <= 0) {
		return;
	}

	const dim3 threads = { n_dims, div_round_up(n_threads_linear, n_dims), 1 };
	const size_t n_threads = threads.x * threads.y;
	const dim3 blocks = { (uint32_t)div_round_up(n_elements * n_dims, n_threads), 1, 1 };

	parallel_for_aos_kernel<<<blocks, threads, shmem_size, stream>>>(
		n_elements, n_dims, fun
	);
}

template <typename F>
inline void parallel_for_gpu_aos(cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
	parallel_for_gpu_aos(0, stream, n_elements, n_dims, std::forward<F>(fun));
}

template <typename F>
inline void parallel_for_gpu_aos(size_t n_elements, uint32_t n_dims, F&& fun) {
	parallel_for_gpu_aos(nullptr, n_elements, n_dims, std::forward<F>(fun));
}

template <typename F>
__global__ void parallel_for_soa_kernel(const size_t n_elements, const uint32_t n_dims, F fun) {
	const size_t elem = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t dim = blockIdx.y;
	if (elem >= n_elements) return;
	if (dim >= n_dims) return;

	fun(elem, dim);
}

template <typename F>
inline void parallel_for_gpu_soa(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
	if (n_elements <= 0 || n_dims <= 0) {
		return;
	}

	const dim3 blocks = { n_blocks_linear(n_elements), n_dims, 1 };

	parallel_for_soa_kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>(
		n_elements, n_dims, fun
	);
}

template <typename F>
inline void parallel_for_gpu_soa(cudaStream_t stream, size_t n_elements, uint32_t n_dims, F&& fun) {
	parallel_for_gpu_soa(0, stream, n_elements, n_dims, std::forward<F>(fun));
}

template <typename F>
inline void parallel_for_gpu_soa(size_t n_elements, uint32_t n_dims, F&& fun) {
	parallel_for_gpu_soa(nullptr, n_elements, n_dims, std::forward<F>(fun));
}
#endif


#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif

/*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
inline std::string prettyDouble(const double val) {
  const double absVal = abs(val);
  char result[1000];
  if      (absVal >= 1e+18f) osp_snprintf(result,1000,"%.1f%c",float(val/1e18f),'E');
  else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",float(val/1e15f),'P');
  else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",float(val/1e12f),'T');
  else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",float(val/1e09f),'G');
  else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",float(val/1e06f),'M');
  else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",float(val/1e03f),'k');
  else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",float(val*1e15f),'f');
  else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",float(val*1e12f),'p');
  else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",float(val*1e09f),'n');
  else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",float(val*1e06f),'u');
  else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",float(val*1e03f),'m');
  else osp_snprintf(result,1000,"%f",(float)val);
  return result;
}

/*! return a nicely formatted number as in "3.4M" instead of
    "3400000", etc, using mulitples of 1024 as in kilobytes,
    etc. Ie, the value 65534 would be 64K, 64000 would be 63.8K */
static std::string prettyBytes(const size_t s)
{
  char buf[1000];
  if (s >= (1024LL*1024LL*1024LL*1024LL)) {
    osp_snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
  } else if (s >= (1024LL*1024LL*1024LL)) {
    osp_snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
  } else if (s >= (1024LL*1024LL)) {
    osp_snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
  } else if (s >= (1024LL)) {
    osp_snprintf(buf, 1000, "%.2fK",s/(1024.f));
  } else {
    osp_snprintf(buf,1000,"%zi",s);
  }
  return buf;
}

/*! return a nicely formatted number as in "3.4M" instead of
    "3400000", etc, using mulitples of thousands (K), millions
    (M), etc. Ie, the value 64000 would be returned as 64K, and
    65536 would be 65.5K */
static std::string prettyNumber(const size_t s)
{
  char buf[1000];
  if (s >= (1000LL*1000LL*1000LL*1000LL)) {
    osp_snprintf(buf, 1000,"%.2fT",s/(1000.f*1000.f*1000.f*1000.f));
  } else if (s >= (1000LL*1000LL*1000LL)) {
    osp_snprintf(buf, 1000, "%.2fG",s/(1000.f*1000.f*1000.f));
  } else if (s >= (1000LL*1000LL)) {
    osp_snprintf(buf, 1000, "%.2fM",s/(1000.f*1000.f));
  } else if (s >= (1000LL)) {
    osp_snprintf(buf, 1000, "%.2fK",s/(1000.f));
  } else {
    osp_snprintf(buf,1000,"%zi",s);
  }
  return buf;
}

static void printUsedGPUMemory(const char* str)
{
  CUDA_SYNC_CHECK();
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  size_t used = total-free;
  printf("%s: %s\n", str, prettyBytes(used).c_str());
}

static void getUsedGPUMemory(unsigned long long* out)
{
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  *out = total-free;
}

inline std::atomic<size_t>& total_n_bytes_allocated() 
{
	static std::atomic<size_t> s_total_n_bytes_allocated{0};
	return s_total_n_bytes_allocated;
}

}
