#include "Sphere.h"

__device__ __host__ Sphere::Sphere() : position({ 0.f, 0.f, 0.f }), radius(0.1f)
{}

__device__ __host__ Sphere::Sphere(const glm::vec3& center, const float radius) : position(center), radius(radius)
{}

__device__ __host__ bool Sphere::hit(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const {
	float a = glm::dot(rayDirection, rayDirection);
	float b = 2.f * glm::dot(rayOrigin - position, rayDirection);
	float c = glm::dot(rayOrigin - position, rayOrigin - position) - radius * radius;
	float del = b * b - 4.f * a * c;
	if (0.f <= del) { // Simplification for intersection.
		return true;
	}
	return false;
}

glm::vec3& Sphere::getPositionRef()
{
	return position;
}

float& Sphere::getRadiusRef()
{
	return radius;
}

void Sphere::setPosition(glm::vec3 pos)
{
	position = pos;
}

void Sphere::setRadius(float r)
{
	radius = r;
}


