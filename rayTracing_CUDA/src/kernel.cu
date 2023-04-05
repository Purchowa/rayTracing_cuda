#include "Kernel.h"

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

static __global__ void sum(uint32_t* d_a, const glm::uvec2 imgDim) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	if (imgDim.x <= x || imgDim.y <= y) {
		return;
	}
	glm::vec2 coord = { (float)x / (float)imgDim.x, (float)y / (float)imgDim.y};
	uint8_t r = (uint8_t)(coord.x * 255.0f);
	uint8_t g = (uint8_t)(coord.y * 255.0f);

	uint32_t index = x + y * blockDim.x * gridDim.x;
	d_a[index] = 0xff000000 | (g << 8) | r;
}

Kernel::Kernel(): kernelTimeMs(0.f), TPB(16){
}

void Kernel::runKernel() {
	// TODO: Jeœli to bêdzie w pêtli siê odœwie¿a³o to warto nie alokowaæ tego za ka¿dym razem
	// TODO: Niepotrzeba bufferSize skoro muszê mieæ imgDim
	uint32_t* d_buffer = nullptr;
	uint32_t bufferSize = imgDim.x * imgDim.y;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (!bufferSize){
		throw std::invalid_argument("CUDA: buffer size is not set!");
	}
	else if (!buffer) {
		throw std::invalid_argument("CUDA: buffer is NULL!");
	}

	gpuErrChk(cudaMalloc(&d_buffer,  bufferSize * sizeof(*d_buffer)));
	gpuErrChk(cudaMemcpy(d_buffer, buffer, bufferSize * sizeof(*d_buffer), cudaMemcpyHostToDevice));

	dim3 gridDim((imgDim.x + TPB - 1) / TPB, (imgDim.y + TPB - 1) / TPB);
	dim3 blockDim(TPB, TPB);

	cudaEventRecord(start);
	sum << < gridDim, blockDim >> > (d_buffer, imgDim);
	gpuErrChk(cudaGetLastError());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernelTimeMs, start, stop);

	gpuErrChk(cudaMemcpy(buffer, d_buffer, bufferSize * sizeof(*d_buffer), cudaMemcpyDeviceToHost));

	gpuErrChk(cudaFree(d_buffer));
}

float Kernel::getKernelTimeMs()
{
	return kernelTimeMs;
}

Kernel::~Kernel() {}

void Kernel::setImgDim(glm::uvec2 imgDim){
	this->imgDim = imgDim;
}

void Kernel::setBuffer(uint32_t* buffer)
{
	this->buffer = buffer;
}
