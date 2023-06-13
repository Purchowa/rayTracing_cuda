#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

/**
* @brief Represents a point of intersection of a single ray and hittable object.
*/
struct HitRecord {
	glm::vec3 normal; /** Normal to hit surface where ray hit */
	glm::vec3 position; /** Point in scene where hit happened */
	float distance; /** Distance from ray origin to hit position */
	int objectIndex; /** Unique index telling which object has been hit */

	/**
	 * @brief All parameter constructor. Also evaluates from which side the ray came. Inside or outside of the object.
	 */
	__device__ HitRecord(const glm::vec3 rayDirection, const glm::vec3 outwardNormal, const glm::vec3 position, const float distance, const int index)
		: position(position), distance(distance), objectIndex(index)
	{
		normal = glm::dot(rayDirection, outwardNormal) > 0.f ? -outwardNormal : outwardNormal;
	}

	__device__ HitRecord() : normal(0.f), position(), distance(-1.f), objectIndex() {}
};
