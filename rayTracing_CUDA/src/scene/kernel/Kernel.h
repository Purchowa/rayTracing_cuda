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
#include "../../app/Settings.h"


using std::string;

/**
* @class Kernel
* @brief Responsible to run kernel directly with parameters. Configure kernel parameters and send values
*/
class Kernel {
public:
	Kernel();
	~Kernel();
	void setImgDim(glm::uvec2 imgDim);
	void setBuffer(uint32_t* buffer, glm::vec3* accColor);
	/**
	 * @brief Main method in which kernel is set up and started.
	 * @param scene - Contains objects that will be rendered on the screen
	 * @param camera - Abstract camera that can move around the scene
	 * @param settings - Instructions for kernel
	 */
	void runKernel(const Scene& scene, const Camera& camera, const Settings settings);
	float getKernelTimeMs();

	uint32_t getAccN() const { return accSampleNum; }
	void setAccN(uint32_t val) { accSampleNum = val; }
private:
	static constexpr int RAY_BOUNCE_COUNT = 5;

	const uint32_t TPB; /*!< Threads per block */
	float kernelTimeMs;
	glm::uvec2 imgDim{ 0, 0 };
	uint32_t* buffer = nullptr; /*!< Buffer to which processed data is passed */
	glm::vec3* accColor = nullptr; /*!< Buffer for accumulating noisy data to create clean image */
	uint32_t accSampleNum = 1; /*!< Sample count used to normalize accumulated color */
};

// CUDA specific
/**
 * @brief Kernel function. Initializes CUDA states that later can be used for random data generation. 
 * @param states - Pointer to array of states
 * @param imgDim - Dimensions of viewport
 * @param seed - seed used to generate random data.
 */
__global__ void initCurand(curandStatePhilox4_32_10_t* states, const glm::uvec2 imgDim, const size_t seed);

/**
 * @biref Main kernel function responsible for tracing rays through every pixel of viewport.
 * @param imgBuff - Output buffer with rendered data.
 * @param accColor - Buffer for accumulating noisy data to create clean image
 * @param accSampleNum - Sample count used to normalize accumulated color
 * @param imgDim - Dimensions of viewport
 * @param rndState - CUDA random states that should be initialized before. They're used for random generation.
 * @param hittable - array of every hittable object in scene.
 * @param hittableSize - number of hittable objects.
 * @param material - list of materials that can be assigned to any hittable object.
 * @param camera - camera that can be moved around scene.
 * @param settings - background color
 */
__global__ void perPixel(uint32_t* imgBuff, glm::vec3 *accColor, const uint32_t accSampleNum, const glm::uvec2 imgDim, curandStatePhilox4_32_10_t* rndState, const Sphere* hittable, const uint32_t hittableSize, const Material* material, const Camera* camera,  Settings* settings);

/**
 * @brief Device function for tracing rays path around scene.
 * @param ray - single ray traveling lonely
 * @param hittable - list of hittable objects in scene
 * @param hittableSize - number of hittable objects in scene.
 * @return HitRecord struct containing necessary information about hit.
 */
__device__ HitRecord traceRay(const Ray ray, const Sphere* hittable, const uint32_t hittableSize);

/**
 * @brief Device function launched for closest hit relative to camera.
 * @param ray - which hit the object
 * @param hitDistance - distance from camera to hit point.
 * @param objectIndex - unique index of object which ray hit.
 * @param hittable - single pointer to sphere which intersected with ray.
 * @return HitRecord with necessary information about intersection.
 */
__device__ HitRecord closestHit(const Ray ray, float hitDistance, int objectIndex, const Sphere* hittable);

/**
 * @brief Device function launched when ray missed every hittable object in scene.
 * @param ray which missed
 * @return HitRecord with distance of -1.f which tells that there was no intersection.
 */
__device__ HitRecord miss(const Ray ray);

/**
 * @brief Device method for generating random direction inside a unit radius sphere.
 * @return random point with length < 1.f
 */
__device__ glm::vec3 randomDirectionUnitSphere(curandStatePhilox4_32_10_t* rndState);

// CUDA Errors
/**
 * @brief Macro for error checks in CUDA operations.
 */
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