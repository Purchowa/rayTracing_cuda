#include "Kernel.h"

__global__ void initCurand(curandStatePhilox4_32_10_t* states, const glm::uvec2 imgDim) 
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t gIndex = x + y * blockDim.x * gridDim.x;

	if (imgDim.x <= x || imgDim.y <= y || imgDim.x * imgDim.y <= gIndex) {
		return;
	}
	curand_init((size_t)gIndex, 0, 0, &states[gIndex]);
	// Sequence 0 and offset 0 for better performance but may result in worse 'randomness'
}

__device__ glm::vec3 randomDirection(curandStatePhilox4_32_10_t* rndState, const glm::vec3& origin) 
{
	auto rndVec3 = [&rndState]() -> glm::vec3 {
		return glm::vec3(2.f * curand_uniform(rndState) - 1.f);
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

	// glm::vec3 closestHit = closestT * ray.direction + shiftOrigin;

	// HitRecord hitRecord(ray.direction, (closestHit - closestSphere->getPosition()) / closestSphere->getRadius()); // normal as unit vector of closestHit so the light is global
	// HitRecord hitRecord(ray.direction, (closestHit - hittable[closestObjIdx].getPosition()) / hittable[closestObjIdx].getRadius(), closestHit, closestT, closestObjIdx );
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

//template <int DEPTH>
//__device__ glm::vec4 colorRaw(const Ray ray, const Sphere* hittable, const uint32_t hittableSize, const glm::vec4& backgroundColor, curandStatePhilox4_32_10_t* rndState)
//{
//	const Sphere* closestSphere = nullptr;
//	Ray nRay = ray;
//	float color{ 1.f };
//	float lightIntensity{1.f};
//	int n = DEPTH;
//	do {
//		glm::vec3 closestShiftOrigin{};
//		float closestT{ FLT_MAX };
//		for (int i = 0; i < hittableSize; i++) {
//			// Shifing current camera to the position of given object. It's used for the calculation of intersections.
//			glm::vec3 shiftOrigin = nRay.origin - hittable[i].getPosition();
//			float t = hittable[i].hit({ shiftOrigin, nRay.direction });
//			if (t < 0.f)
//				continue;
//
//			if (t < closestT) {
//				closestSphere = &hittable[i];
//				closestT = t;
//				closestShiftOrigin = shiftOrigin;
//			}
//		}
//
//		if (closestSphere == nullptr) {
//			return color * backgroundColor;
//		}
//
//		glm::vec3 closestHit = closestT * nRay.direction + closestShiftOrigin;
//
//		//HitRecord hitRecord(nRay.direction, (closestHit - closestSphere->getPosition()) / closestSphere->getRadius()); // normal as unit vector of closestHit so the light is global
//
//		color *= 0.5f;
//		nRay.origin = closestHit;
//		glm::vec3 target = hitRecord.normal + randomDirection(rndState, closestHit);
//		nRay.direction = target;
//		closestSphere = nullptr;
//
//		//glm::vec3 lightSource = glm::normalize(glm::vec3(1.f, 1.f, -1.f));
//		//lightIntensity = glm::max(glm::dot(closestHit, -lightSource), 0.f); // only angles: 0 <= d <= 90
//
//	} while (0 < n--);
//
//	return glm::vec4(0.f, 0.f, 0.f, 1.f);
//	/*return {
//			color.r * lightIntensity,
//			color.g * lightIntensity,
//			color.b * lightIntensity,
//			color.a
//	};*/
//}

__global__ void perPixel(
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
	glm::vec4 backgroundColor = {(1.f - grad) * glm::vec3(1.f, 1.f, 1.f) + grad * glm::vec3(0.5f, 0.7f, 1.0f), 1.f};

	if (!hittableSize) {
		imgBuff[gIndex] = convertFromRGBA(backgroundColor);
		return;
	}

	Ray ray;
	HitRecord hitRecord;
	glm::vec3 lightSource = glm::normalize(glm::vec3(1.f, -1.f, -1.f));
	glm::vec4 sumColor{};

	ray.origin = camera->GetPosition();

	for (int i = 0; i < ANTIALIASING_SAMPLES; i++) {
		glm::vec2 rndCoord{
			( (x + curand_uniform(&rndState[gIndex])) * 2.f ) / float(imgDim.x) - 1.f,
			( (y + curand_uniform(&rndState[gIndex])) * 2.f ) / float(imgDim.y) - 1.f};

		ray.direction = camera->calculateRayDirection(rndCoord);

		hitRecord = traceRay(ray, hittable, hittableSize);
		if (hitRecord.distance < 0.f) { // Didn't hit any hittable
			// imgBuff[gIndex] = convertFromRGBA(backgroundColor);
			sumColor += backgroundColor;
			// continue;
		}
		else {
			float lightIntensity = glm::max(glm::dot(hitRecord.normal, -lightSource), 0.f); // only angles: 0 <= d <= 90
			sumColor += hittable[hitRecord.objectIndex].getColor() * lightIntensity;
		}
			
	}
	
	
	

	glm::vec4 color = sumColor / (float)ANTIALIASING_SAMPLES;
	imgBuff[gIndex] = convertFromRGBA(glm::vec4(
					color.r,
					color.g,
					color.b,
					1.f
			));
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

	perPixel << < gridDim, blockDim >> > (
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
