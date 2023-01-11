#include <string>
#include <vector>
#include <random>

#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <mma.h>

#include <cuMat/src/Errors.h>
#include <tinyformat.h>
#include <third-party/Eigen/Core> // in cuMat

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

std::default_random_engine RND(42);

static void fillRandomHalfMatrix_RowMajor(half* mem, //row-major
	int rows, int cols, bool normalizeRows, bool normalizeCols)
{
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m;
	m.resize(rows, cols);
	std::uniform_real_distribution<float> distr(-1, +1);
	for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
		m(r, c) = distr(RND);
	if (normalizeRows)
		m /= rows;
	if (normalizeCols)
		m /= cols;
	for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
		mem[r + c * rows] = __half2float(m(r, c));
}
static void fillRandomHalfMatrix_ColMajor(half* mem, //row-major
	int rows, int cols, bool normalizeRows, bool normalizeCols)
{
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m;
	m.resize(rows, cols);
	std::uniform_real_distribution<float> distr(-1, +1);
	for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
		m(r, c) = distr(RND);
	if (normalizeRows)
		m /= rows;
	if (normalizeCols)
		m /= cols;
	for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
		mem[c + r * cols] = __half2float(m(r, c));
}

int main()
{
	renderer::VolumeInterpolationNetwork net;
	net.loadNetwork("/home/qadwu/Work/fV-SRN/applications/volnet/results/eval_CompressionTeaser/hdf5/rm60-Hybrid.volnet");

	renderer::GlobalSettings s{};
	// s.scalarType = positions.scalar_type();
	s.volumeShouldProvideNormals = false;
	s.interpolationInObjectSpace = false;
	// const auto oldBoxMax = boxMax();
	// const auto oldBoxMin = boxMin();
	net.setBoxMin(make_double3(0, 0, 0));
	net.setBoxMax(make_double3(1, 1, 1));
	// int channels = outputChannels();

	net.prepareRendering(s);

	std::cout << net.getDefines(s) << std::endl;
	// auto fs = net.getIncludeFileNames(s);
	// for (auto& f : fs) {
	// 	std::cout << f << std::endl;
	// }
	std::cout << net.getConstantDeclarationName(s) << std::endl;
	std::cout << net.getPerThreadType(s) << std::endl;

	void* ptr;
	cudaGetSymbolAddress(&ptr, volumeInterpolationTensorcoresParameters);
	net.fillConstantMemory(s, (CUdeviceptr)ptr, 0);

	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	SRNTestKernel<<<1, BLOCK_SIZE>>>();

	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	// //launch kernel
	// int blockSize;
	// if (s.fixedBlockSize>0)
	// {
	// 	if (s.fixedBlockSize > fun.bestBlockSize())
	// 		throw std::runtime_error("larger block size requested that can be fulfilled");
	// 	blockSize = s.fixedBlockSize;
	// } else
	// {
	// 	blockSize = fun.bestBlockSize();
	// }
	// int minGridSize = std::min(
	// 	int(CUMAT_DIV_UP(batches, blockSize)),
	// 	fun.minGridSize());
	// dim3 virtual_size{
	// 	static_cast<unsigned int>(batches), 1, 1 };
	// bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "IVolumeInterpolation::evaluate", [&]()
	// 	{
	// 		const auto accPosition = accessor< ::kernel::Tensor2Read<scalar_t>>(positions);
	// 		const auto accDirection = hasDirection
	// 			? accessor< ::kernel::Tensor2Read<scalar_t>>(direction)
	// 			: ::kernel::Tensor2Read<scalar_t>();
	// 		const auto accDensity = accessor< ::kernel::Tensor2RW<scalar_t>>(densities);
	// 		const void* args[] = { &virtual_size, &accPosition, &accDirection, &accDensity };
	// 		auto result = cuLaunchKernel(
	// 			fun.fun(), minGridSize, 1, 1, blockSize, 1, 1,
	// 			0, stream, const_cast<void**>(args), NULL);
	// 		if (result != CUDA_SUCCESS)
	// 			return printError(result, kernelName);
	// 		return true;
	// 	});

	return 0;
}