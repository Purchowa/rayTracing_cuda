#pragma once
#include "Ray.h"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct Material {
    glm::vec4 color;
    float roughness;
    float metalic;
    glm::vec3 emmisionColor;
    float emmisionPower;

    Material(glm::vec4 color, float roughness, float metalic,
        glm::vec3 emmissionColor, float emmissionPower)
        : color(color), roughness(roughness), metalic(metalic),
        emmisionColor(emmissionColor), emmisionPower(emmissionPower) {}

    Material() : color(0.5f), roughness(0.f), metalic(0.f), emmisionColor({ 0.0f, 0.0f, 0.0f }), emmisionPower(0.0f) {}

    __device__ __host__ glm::vec3 GetEmmision() const { return emmisionColor * emmisionPower; }
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
