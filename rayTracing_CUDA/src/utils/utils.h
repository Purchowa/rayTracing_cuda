#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

__device__ uint32_t convertFromRGBA(const glm::vec4& color);