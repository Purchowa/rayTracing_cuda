#include "Sphere.h"

__device__ __host__ Sphere::Sphere() : Hittable({ 0.f, 0.f, -1.f }, 0), radius(0.5f)
{}

__device__ __host__ Sphere::Sphere(const glm::vec3& center, const float radius, const int matIndex): Hittable(center, matIndex), radius(radius)
{}

__device__ float Sphere::hit(const Ray& ray) const {
	float a = glm::dot(ray.direction, ray.direction); // >= 0
	float b = 2.f * glm::dot(ray.origin, ray.direction);
	float c = glm::dot(ray.origin, ray.origin) - radius * radius;
	float del = b * b - 4.f * a * c;

	// float t0 = (-b + glm::sqrt(del)) / (2 * a); // backward hit point
	float t1 = (-b - glm::sqrt(del)) / (2.f * a);

	if (del < 0.f) {
		return (float)Hit::NO_HIT;
	}
	return t1;
}

