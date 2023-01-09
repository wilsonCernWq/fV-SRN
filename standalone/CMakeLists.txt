cmake_minimum_required(VERSION 3.10)
project(StandAlone LANGUAGES C CXX CUDA)

cmake_policy(SET CMP0074 NEW) # use _ROOT environment variables
cmake_policy(SET CMP0091 NEW) # set runtime library
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

####################################
# MAIN OPTIONS
####################################
OPTION(RENDERER_ONLY_RESOURCES "Only build the kernel resources, for the UNIX server where the main building is done in setup.py" OFF)
OPTION(RENDERER_SHARED_LIB "Build renderer as a shared library" OFF)
OPTION(RENDERER_BUILD_GUI "Build gui project" OFF)
OPTION(RENDERER_BUILD_TESTS "Build unit test project" OFF)
OPTION(RENDERER_BUILD_BINDINGS "Build the python bindings" OFF)
OPTION(RENDERER_BUILD_CLI "Build command line interface project" OFF)
OPTION(RENDERER_BUILD_OPENGL_SUPPORT "Build with OpenGL rasterization support" OFF)
OPTION(RENDERER_BUILD_COMPRESSION "Build with baseline compression methods for comparison" OFF)
add_definitions(-DKERNEL_DOUBLE_PRECISION=1) # host: double precision for unit tests

####################################
# C++ standard
####################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DNOMINMAX)

####################################
# GENERAL THIRD-PARTY DEPENDENCIES
####################################

# CUDA is always required
find_package(CUDA REQUIRED)
if(COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
	if (WIN32) # inference-gui
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
	else() # server
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 7.5)
	endif()
	STRING(REPLACE ";" "=" ARCH_FLAGS2 "${ARCH_FLAGS}")
	LIST(APPEND CUDA_NVCC_FLAGS "${ARCH_FLAGS2}")
	message(STATUS "cuda flags: ${ARCH_FLAGS2}")
endif()

list(APPEND CUDA_NVCC_FLAGS "-std=c++17" "--use_fast_math" "--generate-line-info" "-Xptxas" "-v")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

set(MY_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
set(CUDA_SEPARABLE_COMPILATION ON)
message(STATUS "final cuda flags: ${CMAKE_CUDA_FLAGS}")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# cuMat-Settings
add_definitions(-DCUMAT_SINGLE_THREAD_CONTEXT=1 -DTHRUST_IGNORE_CUB_VERSION_CHECK=1)

####################################
# create the executable
####################################

set(TEST_FILES 
	# testSRNKernel.cu
	mainSRNKernel.cu

	volume_interpolation_network.cu
)

set(TEST_NAME standalone CACHE INTERNAL "target name for the tests")
add_executable(${TEST_NAME} ${TEST_FILES})
target_include_directories(${TEST_NAME} PRIVATE 
	${CMAKE_SOURCE_DIR}/third-party/tinyformat
	${CMAKE_SOURCE_DIR}/third-party/cuMat
	${CMAKE_SOURCE_DIR}/third-party/cuMat/third-party
	${CMAKE_SOURCE_DIR}/third-party/nlohmann
	${CMAKE_SOURCE_DIR}/third-party/lodepng
	${CMAKE_SOURCE_DIR}/third-party/thread-pool/include
	${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD
	${CMAKE_SOURCE_DIR}/third-party/catch
	${CMAKE_SOURCE_DIR}/third-party/magic_enum/include
)
target_link_libraries(${TEST_NAME}
	${CUDA_LIBRARIES}
	cuda cudadevrt
)
set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 17)
target_compile_definitions(${TEST_NAME} PRIVATE CUMAT_EIGEN_SUPPORT=1)
target_compile_options(${TEST_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
set_target_properties(${TEST_NAME} 
PROPERTIES 
  CUDA_STANDARD 17
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
)
