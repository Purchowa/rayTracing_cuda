#include "Kernel.h"

static __global__ void init_curand(curandStatePhilox4_32_10_t* states, const glm::uvec2 imgDim) {
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t gIndex = x + y * blockDim.x * gridDim.x;

	if (imgDim.x <= x || imgDim.y <= y || imgDim.x * imgDim.y <= gIndex) {
		return;
	}
	curand_init((size_t)gIndex, 0, 0, &states[gIndex]);
	// Sequence 0 and offset 0 for better performance but may result in worse 'randomness'
}

static __device__ glm::vec4 colorRaw(const Ray ray, const Sphere* hittable, const uint32_t hittableSize, const glm::vec4& backgroundColor)
{
	const Sphere* closestSphere = nullptr;
	glm::vec3 closestShiftOrigin{};
	float closestT{ FLT_MAX };
	for (int i = 0; i < hittableSize; i++) {
		// Shifing current camera to the position of given object. It's used for the calculation of intersections.
		glm::vec3 shiftOrigin = ray.origin - hittable[i].getPosition();
		float t = hittable[i].hit({ shiftOrigin, ray.direction });
		if (t < 0.f)
			continue;
		
		if (t < closestT) {
			closestSphere = &hittable[i];
			closestT = t;
			closestShiftOrigin = shiftOrigin;
		}
	}

	if (closestSphere == nullptr) {
		return backgroundColor;
	}

	glm::vec3 closestHit = closestT * ray.direction + closestShiftOrigin;
	
	HitRecord hitRecord(ray.direction, (closestHit - closestSphere->getPosition()) / closestSphere->getRadius()); // normal as unit vector of closestHit so the light is global

	glm::vec3 lightSource = glm::normalize(glm::vec3(1.f, 1.f, -1.f));
	float lightIntensity = glm::max(glm::dot(closestHit, -lightSource), 0.f); // only angles: 0 <= d <= 90

	glm::vec4 color = closestSphere->getColor();


	return glm::vec4(0.5f * (hitRecord.normal + glm::vec3(1.f, 1.f, 1.f) * lightIntensity), 1.f);
	/*return {
			color.r * lightIntensity,
			color.g * lightIntensity,
			color.b * lightIntensity,
			color.a
	};*/
}

static __global__ void trace_ray(
	uint32_t* imgBuff,
	const glm::uvec2 imgDim,
	curandStatePhilox4_32_10_t* rndState,
	const Sphere* hittable,
	const uint32_t hittableSize,
	const Camera* camera) {

	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t gIndex = x + y * blockDim.x * gridDim.x;

	if (imgDim.x <= x || imgDim.y <= y || imgDim.x * imgDim.y <= gIndex) {
		return;
	}
	glm::vec2 coord = {((float)x * 2.f / (float)imgDim.x) - 1.f,
						((float)y * 2.f / (float)imgDim.y) - 1.f}; // [-1; 1]

	float grad = 0.5f * (-coord.y + 1.f);
	glm::vec4 backgroundColor = {(1.f - grad) * glm::vec3(1.f, 1.f, 1.f) +
									grad * glm::vec3(0.5f, 0.7f, 1.0f),
								1.f};

	if (!hittableSize) {
		imgBuff[gIndex] = convertFromRGBA(backgroundColor);
		return;
	}

	Ray ray;
	ray.origin = camera->GetPosition();
	glm::vec4 sumColor{};
	for (int i = 0; i < ANTIALIASING_SAMPLES; i++) {
		glm::vec2 rndCoord{
			( (x + curand_uniform(&rndState[gIndex])) * 2.f ) / float(imgDim.x) - 1.f,
			( (y + curand_uniform(&rndState[gIndex])) * 2.f ) / float(imgDim.y) - 1.f};

		ray.direction = glm::normalize(camera->calculateRayDirection(rndCoord));
		sumColor += colorRaw(ray, hittable, hittableSize, backgroundColor);
	}
	
	glm::vec4 color = sumColor / (float)ANTIALIASING_SAMPLES;
	imgBuff[gIndex] = convertFromRGBA(color);
}


Kernel::Kernel(): kernelTimeMs(0.f), TPB(16){
}

void Kernel::runKernel(const Scene& scene, const Camera& camera) {
	// TODO: Jeœli to bêdzie w pêtli siê odœwie¿a³o to warto nie alokowaæ tego za ka¿dym razem
	uint32_t* d_buffer = nullptr;
	Sphere* d_hittable = nullptr;
	Camera* d_camera = nullptr;
	curandStatePhilox4_32_10_t* d_curandState = nullptr;
	cudaEvent_t start, stop;

	uint32_t bufferSize = imgDim.x * imgDim.y;
	dim3 gridDim((imgDim.x + TPB - 1) / TPB, (imgDim.y + TPB - 1) / TPB);
	dim3 blockDim(TPB, TPB);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (!bufferSize) {
      throw std::invalid_argument("CUDA: buffer size is not set!");
    } else if (!buffer) {
      throw std::invalid_argument("CUDA: buffer is NULL!");
    }

	gpuErrChk(cudaMalloc(&d_buffer,  bufferSize * sizeof(*d_buffer)));
	gpuErrChk(cudaMalloc(&d_hittable, scene.sphere.size() * sizeof(*d_hittable)));
	gpuErrChk(cudaMalloc(&d_curandState, bufferSize * sizeof(*d_curandState)));
	gpuErrChk(cudaMalloc(&d_camera, sizeof(*d_camera)));

	cudaEventRecord(start);

    gpuErrChk(cudaMemcpy(d_buffer, buffer, bufferSize * sizeof(*d_buffer),
                         cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_hittable, scene.sphere.data(),
                         scene.sphere.size() * sizeof(*d_hittable),
                         cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_camera, &camera,
                              sizeof(*d_camera),
                              cudaMemcpyHostToDevice))

	trace_ray << < gridDim, blockDim >> > (
		d_buffer,
		imgDim, d_curandState,
		d_hittable,
		scene.sphere.size(),
		d_camera);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernelTimeMs, start, stop);
	gpuErrChk(cudaGetLastError());

    gpuErrChk(cudaMemcpy(buffer, d_buffer, bufferSize * sizeof(*d_buffer),
                         cudaMemcpyDeviceToHost));

	gpuErrChk(cudaFree(d_buffer));
	gpuErrChk(cudaFree(d_hittable));
	gpuErrChk(cudaFree(d_curandState));
	gpuErrChk(cudaFree(d_camera));
}


  float Kernel::getKernelTimeMs() { return kernelTimeMs; }

  Kernel::~Kernel() {}

  void Kernel::setImgDim(glm::uvec2 imgDim) { this->imgDim = imgDim; }

  void Kernel::setBuffer(uint32_t* buffer) { this->buffer = buffer; }
