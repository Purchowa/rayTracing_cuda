#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

/**
* Represents 
*/
struct HitRecord {
	glm::vec3 normal;
	glm::vec3 position;
	float distance;
	uint32_t objectIndex;

	__device__ HitRecord(glm::vec3 rayDirection, glm::vec3 outwardNormal) {
		normal = glm::dot(rayDirection, outwardNormal) > 0.f ? -outwardNormal : outwardNormal;
	}
};
