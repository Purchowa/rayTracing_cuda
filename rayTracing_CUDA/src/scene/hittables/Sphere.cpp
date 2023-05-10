#include "Sphere.h"

__device__ __host__ Sphere::Sphere() : position({ 0.f, 0.f, 1.f }), radius(0.5f)
{}

__device__ __host__ Sphere::Sphere(const glm::vec3& center, const float radius) : position(center), radius(radius)
{}

__device__ glm::vec4 Sphere::hit(const glm::vec3& rayOrigin, const glm::vec2& pixCoord) const {
	glm::vec3 rayDirection{ pixCoord.x, pixCoord.y, 1.f };

	float a = glm::dot(rayDirection, rayDirection); // >= 0
	float b = 2.f * glm::dot(rayOrigin - position, rayDirection);
	float c = glm::dot(rayOrigin - position, rayOrigin - position) - radius * radius;
	float del = b * b - 4.f * a * c;

	if (del < 0.f) { // Simplification for intersection.
		return { 0.f, 0.f, 0.f, 1.f };
	}

	// float t0 = (-b + glm::sqrt(del)) / (2 * a); // backward hit point
	float t1 = (-b - glm::sqrt(del)) / (2.f * a);

	glm::vec3 closestHitPoint = t1 * rayDirection + rayOrigin;

	return { closestHitPoint, 1.f };
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


