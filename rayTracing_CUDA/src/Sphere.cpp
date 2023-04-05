#include "Sphere.h"
#include <iostream>

Sphere::Sphere(const glm::vec3& center, const float radius): center(center), radius(radius)
{
}

bool Sphere::hit(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const {
	std::cout << center.x << "  -> " << radius << '\n';
	// TODO: Spróbowaæ odpaliæ to na GPU
	return false;
}