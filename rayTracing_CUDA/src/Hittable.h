#pragma once
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuda_runtime.h>

class Hittable {
public:
	__device__ __host__ virtual bool hit(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const = 0;
};