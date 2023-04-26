#include "Kernel.h"

static __global__ void trace_ray(uint32_t* d_a, const glm::uvec2 imgDim, const Sphere* d_hittable, const uint32_t hittableSize) {
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t index = x + y * blockDim.x * gridDim.x;

	if (imgDim.x <= x || imgDim.y <= y || imgDim.x * imgDim.y <= index) {
		return;
	}
	glm::vec2 coord = { ((float)x * 2.f / (float)imgDim.x) - 1.f, ((float)y * 2.f / (float)imgDim.y) - 1.f}; // [-1; 1]
	// coord *= 2.f - 1.f; // [-1; 1] // This doesn't work...

	for (int i = 0; i < hittableSize; i++) {
		if (d_hittable[i].hit({ 0.f, 0.f, 2.f }, { coord.x, coord.y, -1.f })) { // origin of camera, ray direction
			d_a[index] = 0xffabcedf;
			return;
		}
	}
	d_a[index] = 0xff000000;
}


Kernel::Kernel(): kernelTimeMs(0.f), TPB(16){
}

void Kernel::runKernel(Scene& scene) {
	// TODO: Jeœli to bêdzie w pêtli siê odœwie¿a³o to warto nie alokowaæ tego za ka¿dym razem
	uint32_t* d_buffer = nullptr;
	Sphere* d_hittable = nullptr;
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
	gpuErrChk(cudaMalloc(&d_hittable, scene.sphere.size() * sizeof(*d_hittable)));

	gpuErrChk(cudaMemcpy(d_buffer, buffer, bufferSize * sizeof(*d_buffer), cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_hittable, scene.sphere.data(), scene.sphere.size() * sizeof(*d_hittable), cudaMemcpyHostToDevice));

	dim3 gridDim((imgDim.x + TPB - 1) / TPB, (imgDim.y + TPB - 1) / TPB);
	dim3 blockDim(TPB, TPB);

	cudaEventRecord(start);
	trace_ray << < gridDim, blockDim >> > (d_buffer, imgDim, d_hittable, scene.sphere.size());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernelTimeMs, start, stop);
	gpuErrChk(cudaGetLastError());

	gpuErrChk(cudaMemcpy(buffer, d_buffer, bufferSize * sizeof(*d_buffer), cudaMemcpyDeviceToHost));

	gpuErrChk(cudaFree(d_buffer));
	gpuErrChk(cudaFree(d_hittable));
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
