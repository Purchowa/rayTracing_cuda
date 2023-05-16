#pragma once
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <glm/glm.hpp>

#include "../scene/Scene.h"
#include "../utils/utils.h"

using std::string;

class Kernel {
public:
	Kernel();
	~Kernel();
	void setImgDim(glm::uvec2 imgDim);
	void setBuffer(uint32_t* buffer);
	void runKernel(Scene& scene);
	float getKernelTimeMs();

private:
	const uint32_t TPB;
	float kernelTimeMs;
	glm::uvec2 imgDim{ 0, 0 };
	uint32_t* buffer = nullptr;
};

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) {
			cudaDeviceReset(); // destroys all allocations, resets all states
			std::exit(code);
		}
	}
}