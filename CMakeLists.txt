# For optimized version, run with option -DCMAKE_BUILD_TYPE=Release
# To use gprof, add a "-pg" option to end of <set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")>
cmake_minimum_required(VERSION 3.8)

PROJECT(GAMMA_GPU LANGUAGES CXX CUDA)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11")
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -w ${CMAKE_CXX_FLAGS}")

find_package(CUDA REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35;-std=c++11;-O3;-use_fast_math;")


file(GLOB SOURCE src/*.cpp)
file(GLOB CUDA_FILES src/*.cu)
file(GLOB HEADER include/*.h)

include_directories(${CMAKE_SOURCE_DIR}/include)
cuda_add_library(GAMMA_lib ${SOURCE} ${HEADER})
cuda_add_library(CUDA_lib ${CUDA_FILES})
set_target_properties( GAMMA_lib
	PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
cuda_add_executable(GAMMA GAMMA.cu)
set_property(TARGET GAMMA 
	PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(GAMMA GAMMA_lib CUDA_lib)

