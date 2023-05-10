#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

class Sphere{
public:
	__device__ __host__ Sphere();
	__device__ __host__ Sphere(const glm::vec3& center, const float radius);
	__device__ glm::vec4 hit(const glm::vec3& rayOrigin, const glm::vec2& pixCoord) const; // returns color of pixel

	glm::vec3& getPositionRef();
	float& getRadiusRef();

	void setPosition(glm::vec3 pos);
	void setRadius(float r);
private:
	glm::vec3 position;
	float radius;
};
