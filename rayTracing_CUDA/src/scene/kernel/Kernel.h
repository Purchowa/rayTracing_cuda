#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <glm/glm.hpp>
#include <chrono>

#include "../Scene.h"
#include "../utils/utils.h"
#include "../camera/Camera.h"
#include "../hittables/HitRecord.h"

#define ANTIALIASING_SAMPLES 1

using std::string;

class Kernel {
public:
	Kernel();
	~Kernel();
	void setImgDim(glm::uvec2 imgDim);
	void setBuffer(uint32_t* buffer);
	void runKernel(const Scene& scene, const Camera& camera);
	float getKernelTimeMs();

private:
	const uint32_t TPB;
	float kernelTimeMs;
	glm::uvec2 imgDim{ 0, 0 };
	uint32_t* buffer = nullptr;
};

// CUDA specific
__global__ void initCurand(curandStatePhilox4_32_10_t* states, const glm::uvec2 imgDim, const size_t seed);

__global__ void perPixel(uint32_t* imgBuff, const glm::uvec2 imgDim, curandStatePhilox4_32_10_t* rndState, const Sphere* hittable, const uint32_t hittableSize, const Camera* camera);
__device__ glm::vec3 randomDirectionUnitSphere(curandStatePhilox4_32_10_t* rndState);
__device__ HitRecord traceRay(const Ray ray, const Sphere* hittable, const uint32_t hittableSize);
__device__ HitRecord closestHit(const Ray ray, float hitDistance, int objectIndex, const Sphere* hittable);
__device__ HitRecord miss(const Ray ray);


// template <int DEPTH = 50>
// __device__ glm::vec4 colorRaw(const Ray ray, const Sphere* hittable, const uint32_t hittableSize, const glm::vec4& backgroundColor, curandStatePhilox4_32_10_t* rndState);

// Errors
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) {
			cudaDeviceReset(); // destroys all allocations, resets all states
			std::exit(code);
		}
	}
}

#define gpuCuRandErrChk(ans) { cuRandAssert((ans), __FILE__, __LINE__); }
static void cuRandAssert(curandStatus_t code, const char* file, int line, bool abort = true) {
	if (code != CURAND_STATUS_SUCCESS) {
		fprintf(stderr, "GPU curand assert: %d %s %d\n", code, file, line); // int error code
		if (abort) {
			cudaDeviceReset(); // destroys all allocations, resets all states
			std::exit(code);
		}
	}
}