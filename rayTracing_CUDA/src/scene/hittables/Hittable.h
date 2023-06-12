#pragma once
#include "Ray.h"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct Material {
    glm::vec4 color;
    float roughness;
    float metalic;
    float emmisionPower;

    Material(glm::vec4 color, float roughness, float metalic, float emmissionPower)
        : color(color), roughness(roughness), metalic(metalic),
          emmisionPower(emmissionPower) {}

    Material() : color(0.5f), roughness(0.f), metalic(0.f), emmisionPower(0.0f) {}

    __device__ __host__ glm::vec3 getEmmision() const { return color * emmisionPower; }
};

class Hittable {
public:
    const enum class Hit { NO_HIT = -1 };

protected:
    Hittable(glm::vec3 position, int matIdx)
        : position(position), materialIndex(matIdx) {}
    glm::vec3 position;

    int materialIndex;
};
