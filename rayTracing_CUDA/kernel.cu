#include "Kernel.h"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) {
			std::exit(code);
		}
	}
}

static __global__ void sum(uint32_t* d_a, const uint32_t N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < N) {
		d_a[index] = index;
		d_a[index] |= 0xff000000;
	}
}

Kernel::Kernel(): kernelTimeMs(0.f), TPB(64){
}

void Kernel::runKernel() {
	// TODO: Jeœli to bêdzie w pêtli siê odœwie¿a³o to warto nie alokowaæ tego za ka¿dym razem
	uint32_t* d_buffer = nullptr;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (!bufferSize){
		throw std::invalid_argument("buffer_size is NULL!");
	}
	else if (!buffer) {
		throw std::invalid_argument("buffer is NULL!");
	}

	gpuErrChk(cudaMalloc(&d_buffer,  bufferSize * sizeof(*d_buffer)));

	gpuErrChk(cudaMemcpy(d_buffer, buffer, bufferSize * sizeof(*d_buffer), cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	sum <<< (bufferSize + TPB - 1) / TPB, TPB >>> (d_buffer, bufferSize);
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

void Kernel::setBufferSize(uint32_t size){
	this->bufferSize = size;
}

void Kernel::setBuffer(uint32_t* buffer)
{
	this->buffer = buffer;
}
