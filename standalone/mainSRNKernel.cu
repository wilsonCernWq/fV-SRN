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

#define BLOCK_SIZE 512
#define NUM_HIDDEN_LAYERS 0
#define HIDDEN_CHANNELS_DIV16 2
#define HAS_FOURIER_FEATURES 1
#define NUM_FOURIER_FEATURES ((HIDDEN_CHANNELS_DIV16*16-4)/2)
#define USE_DIRECTION 0 
#define ACTIVATION ReLU
#define OUTPUT_MODE OUTPUT_MODE_DENSITY
#define FIRST_AND_LAST_IN_SHARED_MEMORY 0
#define LATENT_GRID_CHANNELS_DIV16 1
#define LATENT_GRID_ENCODING LATENT_GRID_ENCODING_FLOAT // The encoding for the latent grid
#define PASS_TIME_TO_NETWORK 0
#define GRADIENT_MODE GRADIENT_MODE_OFF_OR_DIRECT // the gradient computation mode

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
	VolumeInterpolationTensorcores srn;
#if OUTPUT_IS_COLOR==1
	auto out = srn.eval<real4>(make_real2(idx, 0, 0), dummy_direction, 0);
	printf("[%04d] -> r=%.4f, g=%.4f, b=%.4f, a=%.4f\n",
		idx, out.value.x, out.value.y, out.value.z, out.value.w);
#else
	auto out = srn.eval<real_t>(make_real3(idx, 0, 0), dummy_direction, 0);
	printf("[%04d] -> d=%.4f\n",
		idx, out.value);
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
	VolumeInterpolationTensorcoresParameters p;
#if HAS_FOURIER_FEATURES==1
	fillRandomHalfMatrix_ColMajor(p.cWeightsFourier, 3, NUM_FOURIER_FEATURES, false, true);
#else
	fillRandomHalfMatrix_ColMajor(p.cWeightsFirst, 3, HIDDEN_CHANNELS, false, true);
	fillRandomHalfMatrix_ColMajor(p.cBiasFirst,    1, HIDDEN_CHANNELS, false, true);
#endif
	fillRandomHalfMatrix_RowMajor(p.cWeightsHidden, HIDDEN_CHANNELS, HIDDEN_CHANNELS * NUM_HIDDEN_LAYERS, false, true);
	fillRandomHalfMatrix_RowMajor(p.cBiasHidden, HIDDEN_CHANNELS, NUM_HIDDEN_LAYERS, false, false);
	//for (int i = 0; i < HIDDEN_CHANNELS * NUM_HIDDEN_LAYERS; ++i)
	//	p.cBiasHidden[i] = __float2half(i);
#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT)
	fillRandomHalfMatrix_ColMajor(reinterpret_cast<half*>(p.cWeightsLast), 4, HIDDEN_CHANNELS, false, true);
	fillRandomHalfMatrix_ColMajor(reinterpret_cast<half*>(p.cBiasLast), 4, 1, false, false);
#else
	fillRandomHalfMatrix_ColMajor(p.cWeightsLast, 1, HIDDEN_CHANNELS, false, true);
	fillRandomHalfMatrix_ColMajor(&p.cBiasLast, 1, 1, false, false);
#endif

	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(volumeInterpolationTensorcoresParameters,
		&p, sizeof(VolumeInterpolationTensorcoresParameters)));

	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	SRNTestKernel<<<1, BLOCK_SIZE>>>();

	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	return 0;
}