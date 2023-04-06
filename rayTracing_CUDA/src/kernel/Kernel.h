#pragma once
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <glm/glm.hpp>
#include "../hittables/Sphere.h"

using std::string;

class Kernel {
public:
	Kernel::Kernel();
	Kernel::~Kernel();
	void setImgDim(glm::uvec2 imgDim);
	void setBuffer(uint32_t* buffer);
	void runKernel();
	float getKernelTimeMs();

private:
	const uint32_t TPB;
	float kernelTimeMs;
	glm::uvec2 imgDim{ 0, 0 };
	uint32_t* buffer = nullptr;
};
