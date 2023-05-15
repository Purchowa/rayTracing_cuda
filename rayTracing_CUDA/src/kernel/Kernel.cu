#include "Kernel.h"

static __global__ void trace_ray(uint32_t* d_imgBuff, const glm::uvec2 imgDim, const Sphere* d_hittable, const uint32_t hittableSize) {
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t gIndex = x + y * blockDim.x * gridDim.x;

	if (imgDim.x <= x || imgDim.y <= y || imgDim.x * imgDim.y <= gIndex) {
		return;
	}
	glm::vec2 coord = { ((float)x * 2.f / (float)imgDim.x) - 1.f, ((float)y * 2.f / (float)imgDim.y) - 1.f}; // [-1; 1]

	if (!hittableSize) {
		d_imgBuff[gIndex] = convertFromRGBA({0.f, 0.f, 0.f, 1.f});
		return;
	}

	const glm::vec3 rayOrigin = { 0.f, 0.f, 1.f };
	glm::vec3 rayDirection = glm::normalize(glm::vec3(coord.x, coord.y, -1.f));

	const Sphere* closestSphere = nullptr;
	glm::vec3 closestShiftOrigin{};
	float closestT{FLT_MAX};

	for (int i = 0; i < hittableSize; i++) {
		// Shifing current camera to the position of given object. It's for the calculation of intersections.
		glm::vec3 shiftOrigin = rayOrigin - d_hittable[i].getPosition();
		float t = d_hittable[i].hit(shiftOrigin, rayDirection);
		if (t < 0.f)
			continue;

		if (t < closestT) {
			closestSphere = &d_hittable[i];
			closestT = t;
			closestShiftOrigin = shiftOrigin;
		}
	}

	if (closestSphere == nullptr) {
		d_imgBuff[gIndex] = convertFromRGBA({0.f, 0.f, 0.f, 1.f});
		return;
	}

	glm::vec3 closestHit = closestT * rayDirection + closestShiftOrigin;
	glm::vec3 normal = glm::normalize(closestHit); // normal as unit vector of closestHit

	glm::vec3 lightSource = glm::normalize(glm::vec3(1.f, 1.f, -1.f));
	float lightIntensity = glm::max(glm::dot(normal, -lightSource), 0.f); // only angles: 0 <= d <= 90

	d_imgBuff[gIndex] = convertFromRGBA(
		{
			closestSphere->getColor().r * lightIntensity,
			closestSphere->getColor().g * lightIntensity,
			closestSphere->getColor().b * lightIntensity,
			closestSphere->getColor().a 
		});
	// d_imgBuff[gIndex] = convertFromRGBA(closestSphere->getColor() * lightIntensity);
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
