#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

class Sphere{
public:
	__device__ __host__ Sphere();
	__device__ __host__ Sphere(const glm::vec3& center, const float radius);
	__device__ __host__ bool hit(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) const;

	glm::vec3& getPositionRef();
	float& getRadiusRef();

	void setPosition(glm::vec3 pos);
	void setRadius(float r);
private:
	glm::vec3 position;
	float radius;
};
