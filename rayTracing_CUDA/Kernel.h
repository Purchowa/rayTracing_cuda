#pragma once
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <glm/glm.hpp>

using std::string;

class Kernel {
public:
	Kernel::Kernel();
	Kernel::~Kernel();
	void setBufferSize(uint32_t size);
	void setBuffer(uint32_t* buffer);
	void runKernel();
	float getKernelTimeMs();

private:
	const int TPB;
	float kernelTimeMs;
	uint32_t bufferSize = NULL;
	uint32_t* buffer = nullptr;
};
