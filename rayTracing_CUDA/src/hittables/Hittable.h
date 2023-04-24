#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include <iostream>
class Hittable {
public:
	__device__ __host__ virtual bool hit(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const = 0;
};