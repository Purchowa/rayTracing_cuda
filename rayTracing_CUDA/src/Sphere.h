#pragma once
#include "Hittable.h"

class Sphere : public Hittable {
public:
	__device__ __host__ Sphere(const glm::vec3& center, const float radius);
	__device__ __host__ virtual bool hit(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const override;

private:
	glm::vec3 center;
	float radius;
};