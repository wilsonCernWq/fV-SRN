#include <string>
#include <vector>
#include <random>

#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <mma.h>

#include "loader.hpp"
#include "cuda_buffer.h"

#include "volume_interpolation_network.h"

// ------------- some definitions --------------- //

#define OUTPUT_MODE_DENSITY 0                  // real_t output
#define OUTPUT_MODE_DENSITY_DIRECT 1           // real_t output
#define OUTPUT_MODE_RGBO 2                     // real4  output
#define OUTPUT_MODE_RGBO_DIRECT 3              // real4  output
#define OUTPUT_MODE_DENSITY_GRADIENT 4         // real_t  + gradient output
#define OUTPUT_MODE_DENSITY_GRADIENT_DIRECT 5  // real_t  + gradient output
#define OUTPUT_MODE_DENSITY_GRADIENT_CUBIC 6   // real_t  + gradient output, but gradients are cubed
#define OUTPUT_MODE_DENSITY_CURVATURE 7        // density + gradient + curvature
#define OUTPUT_MODE_DENSITY_CURVATURE_DIRECT 8 // density + gradient + curvature

#define LATENT_GRID_ENCODING_FLOAT 0         // direct float storage
#define LATENT_GRID_ENCODING_BYTE_LINEAR 1   // bytes, linear storage
#define LATENT_GRID_ENCODING_BYTE_GAUSSIAN 2 // bytes, gaussian mapping

#define GRADIENT_MODE_OFF_OR_DIRECT 0
#define GRADIENT_MODE_FINITE_DIFFERENCES 1
#define GRADIENT_MODE_ADJOINT_METHOD 2

// ------------- config --------------- //

// #define BLOCK_SIZE 512
// #define NUM_HIDDEN_LAYERS 0
// #define HIDDEN_CHANNELS_DIV16 2
// #define HAS_FOURIER_FEATURES 1
// #define NUM_FOURIER_FEATURES ((HIDDEN_CHANNELS_DIV16*16-4)/2)
// #define USE_DIRECTION 0 
// #define ACTIVATION ReLU
// #define OUTPUT_MODE OUTPUT_MODE_DENSITY
// #define FIRST_AND_LAST_IN_SHARED_MEMORY 0
// #define LATENT_GRID_CHANNELS_DIV16 1
// #define LATENT_GRID_ENCODING LATENT_GRID_ENCODING_FLOAT // The encoding for the latent grid
// #define PASS_TIME_TO_NETWORK 0
// #define GRADIENT_MODE GRADIENT_MODE_OFF_OR_DIRECT // the gradient computation mode

#define BLOCK_SIZE 256
#define NUM_HIDDEN_LAYERS 2
#define HIDDEN_CHANNELS_DIV16 2
#define HAS_FOURIER_FEATURES 1
#define NUM_FOURIER_FEATURES 14
#define USE_DIRECTION 0
#define ACTIVATION SnakeAlt
#define OUTPUT_MODE 1
#define FIRST_AND_LAST_IN_SHARED_MEMORY 0
#define LATENT_GRID_CHANNELS_DIV16 1
#define LATENT_GRID_ENCODING 0
#define PASS_TIME_TO_NETWORK 0
#define GRADIENT_MODE 0

// ------------- looks like some verifications --------------- //

#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT)
#define OUTPUT_IS_COLOR 1
#else
#define OUTPUT_IS_COLOR 0
#endif

#define MIN_ONE(val) (((val)>0)?(val):(1))

#include "renderer_volume_tensorcores.cuh"

// ------------- start --------------- //

typedef kernel::VolumeInterpolationTensorcoresParameters VolumeInterpolationTensorcoresParameters;
typedef kernel::VolumeInterpolationTensorcores VolumeInterpolationTensorcores;

#define dummy_direction make_real3(0, 0, 0)

__global__ void SRNTestKernel()
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int tot = blockDim.x * gridDim.x;

	VolumeInterpolationTensorcores srn;
#if OUTPUT_IS_COLOR==1
	auto out = srn.eval<real4>(make_real2(idx, 0, 0), dummy_direction, 0);
	printf("[%04d] -> r=%.4f, g=%.4f, b=%.4f, a=%.4f\n",
		idx, out.value.x, out.value.y, out.value.z, out.value.w);
#else
	auto out = srn.eval<real_t>(make_real3(idx/(float)tot, 0, 0), dummy_direction, 0);
	printf("[%04d] -> d=%.4f\n", idx, out.value);
#endif
}

#if OUTPUT_IS_COLOR==1
#error "wrong output fomat"
#endif

__global__ void SRNTestDecode(int dimx, int dimy, int dimz, float* __restrict__ values)
{
  	const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

	const uint64_t N = (uint64_t)dimx * (uint64_t)dimy * (uint64_t)dimz;
  	if (i >= N) return;

	const uint64_t stride = (uint64_t)dimx * dimy;
	const int32_t x =  i % dimx;
	const int32_t y = (i % stride) / dimx;
	const int32_t z =  i / stride;

	VolumeInterpolationTensorcores srn;

	auto in  = make_real3((x+0.5f) / dimx, (y+0.5f) / dimy, (z+0.5f) / dimz);
	auto out = srn.eval<real_t>(in, dummy_direction, 0);
	values[i] = out.value;
}

int main()
{
	GridDesc data = loader::load("/home/qadwu/Work/fV-SRN/applications/volumes/RichtmyerMeshkov/ppm-t0060.cvol");
	std::cout << "data.dims = " << data.dims[0] << " " << data.dims[1] << " " << data.dims[2] << std::endl;
	std::cout << "data.type = " << data.type << std::endl;
	std::cout << std::endl;

	const uint64_t n_voxels = (uint64_t)data.dims[0] * (uint64_t)data.dims[1] * (uint64_t)data.dims[2];

	CUDABuffer gmem;
	gmem.resize(n_voxels * sizeof(float));

	renderer::VolumeInterpolationNetwork net;
	net.loadNetwork("/home/qadwu/Work/fV-SRN/applications/volnet/results/eval_CompressionTeaser/hdf5/rm60-Hybrid.volnet");

	renderer::GlobalSettings s{};
	s.volumeShouldProvideNormals = false;
	s.interpolationInObjectSpace = false;
	net.setBoxMin(make_double3(0, 0, 0));
	net.setBoxMax(make_double3(1, 1, 1));
	net.prepareRendering(s);

	std::cout << "-------" << std::endl;
	std::cout << net.getDefines(s) << std::endl;
	std::cout << "-------" << std::endl;
	std::cout << net.getConstantDeclarationName(s) << std::endl;
	std::cout << net.getPerThreadType(s) << std::endl;
	std::cout << "-------" << std::endl;

	void* ptr;
	cudaGetSymbolAddress(&ptr, volumeInterpolationTensorcoresParameters);
	net.fillConstantMemory(s, (CUdeviceptr)ptr, 0);

	CUDA_CHECK(cudaDeviceSynchronize());

	// SRNTestKernel<<<1, BLOCK_SIZE>>>();

	SRNTestDecode<<<util::div_round_up<uint64_t>(n_voxels, BLOCK_SIZE), BLOCK_SIZE>>>(
		data.dims[0], data.dims[1], data.dims[2],
		(float*)gmem.d_pointer()
	);

	CUDA_CHECK(cudaDeviceSynchronize());

	std::vector<float> output(n_voxels);
	gmem.download(output.data(), n_voxels);

    auto w = vidi::filemap_write_create("test.raw", n_voxels * sizeof(float));
    vidi::filemap_random_write(w, 0, output.data(), n_voxels * sizeof(float));
    vidi::filemap_close(w);

	return 0;
}