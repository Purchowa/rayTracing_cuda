#include "utils.h"
#pragma once

__device__ uint32_t convertFromRGBA(const glm::vec3& color) // converts unsigned int color to float type
{
	glm::vec3 colorClmp = glm::clamp(color, 0.f, 1.f);

	return (
		((uint32_t)(255.f) << 24) // Can't change alpha channel
		| ((uint32_t)(colorClmp.b * 255.f) << 16)
		| ((uint32_t)(colorClmp.g * 255.f) << 8)
		| ((uint32_t)(colorClmp.r * 255.f)));
}

__device__ bool nearZero(glm::vec3 ray)
{
	const auto eps = 1e-7;
	if (fabs(ray.x) < eps && fabs(ray.y) < eps && fabs(ray.z) < eps) {
		return true;
	}
	return false;
}
