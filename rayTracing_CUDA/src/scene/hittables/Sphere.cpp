#include "Sphere.h"

__device__ __host__ Sphere::Sphere() : Hittable({ 0.f, 0.f, -1.f }, {0.5f, 0.5, 0.f, 1.f}), radius(0.5f)
{}

__device__ __host__ Sphere::Sphere(const glm::vec3& center, const glm::vec4& color, const float radius): Hittable(center, color), radius(radius)
{}

__device__ float Sphere::hit(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const {
	float a = glm::dot(rayDirection, rayDirection); // >= 0
	float b = 2.f * glm::dot(rayOrigin, rayDirection);
	float c = glm::dot(rayOrigin, rayOrigin) - radius * radius;
	float del = b * b - 4.f * a * c;

	// float t0 = (-b + glm::sqrt(del)) / (2 * a); // backward hit point
	float t1 = (-b - glm::sqrt(del)) / (2.f * a);

	if (del < 0.f) {
		return (float)Hit::NO_HIT;
	}
	return t1;
}

__device__ glm::vec4 Sphere::getColor() const {
	return color;
}

__device__ glm::vec3 Sphere::getPosition() const
{
	return position;
}

glm::vec3& Sphere::getPositionRef()
{
	return position;
}

float& Sphere::getRadiusRef()
{
	return radius;
}

glm::vec4& Sphere::getColorRef() {
	return color;
}

