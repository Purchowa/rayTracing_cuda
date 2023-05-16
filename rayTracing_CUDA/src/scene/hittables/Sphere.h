#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "Hittable.h"

class Sphere : private Hittable{
public:
	__device__ __host__ Sphere();
	__device__ __host__ Sphere(const glm::vec3& center, const glm::vec4& color, const float radius);

	/**
	* Calculates the closer hit point of given square using closest multiplier 't'. Calculations are done using sphere formula.
	* @param Ray which containts origin of ray (camera origin) and direction for individual pixel.
	* @return closest hit point
	*/
	__device__ float hit(const Ray& ray) const;
	__device__ glm::vec4 getColor() const;
	__device__ glm::vec3 getPosition() const;

	glm::vec3& getPositionRef();
	float& getRadiusRef();
	glm::vec4& getColorRef();

private:
	float radius;
};
