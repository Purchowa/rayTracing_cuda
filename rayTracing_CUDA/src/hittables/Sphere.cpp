#include "Sphere.h"

__device__ __host__ Sphere::Sphere() : center({ 0.f, 0.f, 0.f }), radius(0.f)
{}

__device__ __host__ Sphere::Sphere(const glm::vec3& center, const float radius) : center(center), radius(radius)
{}

__device__ __host__ bool Sphere::hit(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const {
	float a = glm::dot(rayDirection, rayDirection);
	float b = 2 * glm::dot(rayOrigin - center, rayDirection);
	float c = glm::dot(rayOrigin - center, rayOrigin - center) - radius * radius;
	float del = b * b - 4 * a * c;
	if (0 <= del) { // Simplification for intersection.
		return true;
	}
	return false;
}