#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

/**
* Represents 
*/
struct HitRecord {
	glm::vec3 normal;

	__device__ HitRecord(glm::vec3 rayDirection, glm::vec3 outwardNormal) {
		normal = glm::dot(rayDirection, outwardNormal) > 0.f ? -outwardNormal : outwardNormal;
	}
};
