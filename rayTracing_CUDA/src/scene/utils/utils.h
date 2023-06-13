#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

/**
 * @brief Converts floating type colors (0.f -> 1.f) to 8 bit per channel color model (RGB). Alpha is always set to 255.
 * @return 32 bit color where every 8 bits represent specific color channel RGBA.
 */
__device__ uint32_t convertFromRGBA(const glm::vec3& color);

__device__ bool nearZero(glm::vec3 ray);