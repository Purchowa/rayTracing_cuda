#pragma once
#include "Hittable.h"

class Sphere : public Hittable {
public:
	__device__ __host__ Sphere() : center({ 0.f, 0.f, 0.f }), radius(0.f)
	{}
	__device__ __host__ Sphere(const glm::vec3& center, const float radius) : center(center), radius(radius)
	{};
	__device__ __host__ bool hit(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const override {
		float a = glm::dot(rayDirection, rayDirection);
		float b = 2 * glm::dot(rayOrigin - center, rayDirection);
		float c = glm::dot(rayOrigin - center, rayOrigin - center) - radius * radius;
		float del = b * b - 4.f * a * c;
		if (0 <= del) { // Simplification for intersection.
			return true;
		}
		return false;
	}
private:
	glm::vec3 center;
	float radius;
};
