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
	int objectIndex;

	__device__ HitRecord(const glm::vec3 rayDirection, const glm::vec3 outwardNormal, const glm::vec3 position, const float distance, const int index)
		: position(position), distance(distance), objectIndex(index)
	{
		normal = glm::dot(rayDirection, outwardNormal) > 0.f ? -outwardNormal : outwardNormal;
	}

	__device__ HitRecord() : normal(0.f), position(), distance(-1.f), objectIndex() {}
};
