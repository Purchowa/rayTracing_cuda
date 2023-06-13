#include "Kernel.h"

__global__ void initCurand(curandStatePhilox4_32_10_t* states, const glm::uvec2 imgDim, const size_t seed)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t gIndex = x + y * blockDim.x * gridDim.x;

	if (imgDim.x <= x || imgDim.y <= y || imgDim.x * imgDim.y <= gIndex) {
		return;
	}
	curand_init(seed, (size_t)gIndex, 0, &states[gIndex]);
	// Sequence 0 and offset 0 for better performance but may result in worse 'randomness'
}

__device__ glm::vec3 randomDirectionUnitSphere(curandStatePhilox4_32_10_t* rndState) 
{
	auto rndVec3 = [&rndState]() -> glm::vec3 {
		return 2.f * glm::vec3(curand_uniform(rndState), curand_uniform(rndState), curand_uniform(rndState)) - 1.f;
	};
	glm::vec3 randomPoint = rndVec3();
	while (1.f <= glm::length(randomPoint)) {
		randomPoint = rndVec3();
	}
	return randomPoint;
}

__device__ HitRecord traceRay(const Ray ray, const Sphere* hittable, const uint32_t hittableSize)
{
	int closestObjIdx = -1;
	float closestT{ FLT_MAX };
	glm::vec3 shiftOrigin{};

	for (int i = 0; i < hittableSize; i++) {
		// Shifing current camera to the position of given object. It's used for the calculation of intersections.
		shiftOrigin = ray.origin - hittable[i].getPosition();
		float t = hittable[i].hit({ shiftOrigin, ray.direction });
		if (t < 0.f)
			continue;

		if (t < closestT) {
			closestObjIdx = i;
			closestT = t;
		}
	}

	if (closestObjIdx < 0) {
		return miss(ray);
	}
	return closestHit(ray, closestT, closestObjIdx, hittable);
}


__device__ HitRecord closestHit(const Ray ray, float hitDistance, int objectIndex, const Sphere* hittable)
{
	const Sphere& closestSphere = hittable[objectIndex];
	glm::vec3 origin = ray.origin - closestSphere.getPosition(); // Move back to the origin
	
	glm::vec3 hitPoint = origin + ray.direction * hitDistance;
	glm::vec3 normal = glm::normalize(hitPoint);

	hitPoint += closestSphere.getPosition(); // Move into real position

	HitRecord hitRecord(ray.direction, normal, hitPoint, hitDistance, objectIndex);
	return hitRecord;
}


__device__ HitRecord miss(const Ray ray)
{
	return HitRecord();
}
template <int ANTIALIASING_SAMPLES, int RAY_BOUNCE_COUNT>
__global__ void perPixel(
	uint32_t* imgBuff,
	glm::vec3* accColor,
	const uint32_t accSampleNum,
	const glm::uvec2 imgDim,
	curandStatePhilox4_32_10_t* rndState,
	const Sphere* hittable,
	const uint32_t hittableSize,
	const Material* material,
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
	glm::vec3 backgroundColor = {(1.f - grad) * glm::vec3(1.f, 1.f, 1.f) + grad * glm::vec3(0.3, 0.4, 0.5)};
	// backgroundColor = glm::vec4(0.f, 0.f, 0.f, 1.f);

	if (!hittableSize) {
		imgBuff[gIndex] = convertFromRGBA(glm::vec4(backgroundColor, 1.f));
		return;
	}

	Ray ray;
	HitRecord hitRecord;
	glm::vec3 light{0.f};
	glm::vec3 attenuation{ 1.f };

	ray.origin = camera->GetPosition();
	
	for (int i = 0; i < ANTIALIASING_SAMPLES; i++) {
		glm::vec2 rndCoord{
			( (x + curand_uniform(&rndState[gIndex])) * 2.f ) / float(imgDim.x) - 1.f,
			( (y + curand_uniform(&rndState[gIndex])) * 2.f ) / float(imgDim.y) - 1.f};

		ray.direction = camera->calculateRayDirection(rndCoord);
		for (int j = 0; j < RAY_BOUNCE_COUNT; j++){
			hitRecord = traceRay(ray, hittable, hittableSize);
			const Sphere* sphere = &hittable[hitRecord.objectIndex];
			const Material* mat = &material[sphere->getMaterialIdx()];

			if (hitRecord.distance < 0.f) { // Didn't hit any hittable
				light += backgroundColor * attenuation;
				break;
			}
			light += mat->getEmissionPower();
			attenuation *= mat->color;

			ray.origin = hitRecord.position + hitRecord.normal * 0.0001f;
			ray.direction = hitRecord.normal + mat->roughness * randomDirectionUnitSphere(&rndState[gIndex]);
		}
	}
	glm::vec3 color = light / (float)ANTIALIASING_SAMPLES;

	uint32_t& buff = imgBuff[gIndex];
	glm::vec3& acc = accColor[gIndex];
	glm::vec3 currAcc{acc};

	if (accSampleNum <= 1) {
		acc = color;
	}
	else {
		acc += color;
		currAcc = acc / glm::vec3(accSampleNum);
	}
	buff = convertFromRGBA(currAcc);
}


Kernel::Kernel(): kernelTimeMs(0.f), TPB(16){
}

void Kernel::runKernel(const Scene& scene, const Camera& camera, const Settings settings) {
	uint32_t* d_buffer = nullptr;
	glm::vec3* d_accColor = nullptr;
	Sphere* d_hittable = nullptr;
	Material* d_material = nullptr;
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
    } 
	else if (!buffer) {
		throw std::invalid_argument("CUDA: buffer is NULL!");
    }
	else if (!accColor) {
		throw std::invalid_argument("CUDA: accColor buffer is NULL!");
	}

	gpuErrChk(cudaMalloc(&d_buffer,  bufferSize * sizeof(*d_buffer)));
	gpuErrChk(cudaMalloc(&d_accColor, bufferSize * sizeof(*d_accColor)));

	gpuErrChk(cudaMalloc(&d_hittable, scene.sphere.size() * sizeof(*d_hittable)));
	gpuErrChk(cudaMalloc(&d_material, scene.material.size() * sizeof(*d_material)));
	gpuErrChk(cudaMalloc(&d_curandState, bufferSize * sizeof(*d_curandState)));
	gpuErrChk(cudaMalloc(&d_camera, sizeof(*d_camera)));
	cudaEventRecord(start);

	auto duration = std::chrono::system_clock::now().time_since_epoch();
	initCurand << < gridDim, blockDim >> > (d_curandState, imgDim, size_t(duration.count()));

    gpuErrChk(cudaMemcpy(d_hittable, scene.sphere.data(),
                         scene.sphere.size() * sizeof(*d_hittable),
                         cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_accColor, accColor, bufferSize * sizeof(*d_accColor),
						 cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_material, scene.material.data(), 
						 scene.material.size() * sizeof(*d_material),
						 cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_camera, &camera,
		sizeof(*d_camera),
		cudaMemcpyHostToDevice))


	if (camera.Moved() || !settings.accumulate)
		accSampleNum = 1;
	else
		accSampleNum++;

	perPixel<ANTIALIASING_SAMPLES, RAY_BOUNCE_COUNT> << < gridDim, blockDim >> > (
		d_buffer,
		d_accColor,
		accSampleNum,
		imgDim, d_curandState,
		d_hittable,
		scene.sphere.size(),
		d_material,
		d_camera);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernelTimeMs, start, stop);
	gpuErrChk(cudaGetLastError());

    gpuErrChk(cudaMemcpy(buffer, d_buffer, bufferSize * sizeof(*d_buffer),
                         cudaMemcpyDeviceToHost));
	gpuErrChk(cudaMemcpy(accColor, d_accColor, bufferSize * sizeof(*d_accColor),
		cudaMemcpyDeviceToHost));

	gpuErrChk(cudaFree(d_buffer));
	gpuErrChk(cudaFree(d_accColor));
	gpuErrChk(cudaFree(d_hittable));
	gpuErrChk(cudaFree(d_material));
	gpuErrChk(cudaFree(d_curandState));
	gpuErrChk(cudaFree(d_camera));
	gpuErrChk(cudaGetLastError());
}


  float Kernel::getKernelTimeMs() { return kernelTimeMs; }

  Kernel::~Kernel() {}

  void Kernel::setImgDim(glm::uvec2 imgDim) { this->imgDim = imgDim; }

  void Kernel::setBuffer(uint32_t* buffer, glm::vec3* accColor) {
	  this->buffer = buffer;
	  this->accColor = accColor;
  }
