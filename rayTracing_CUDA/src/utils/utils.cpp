#include "utils.h"
#pragma once

__device__ uint32_t convertFromRGBA(const glm::vec4& color) // converts unsigned int color to float type
{
	glm::vec4 colorClmp = glm::clamp(color, 0.f, 1.f);

	return (
		((uint32_t)(colorClmp.a * 255.f) << 24)
		| ((uint32_t)(colorClmp.b * 255.f) << 16)
		| ((uint32_t)(colorClmp.g * 255.f) << 8)
		| ((uint32_t)(colorClmp.r * 255.f)));
}
