#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "Hittable.h"

/**
* @brief Defines 3D sphere on scene.
*/
class Sphere : private Hittable{
public:
	__device__ __host__ Sphere();
	__device__ __host__ Sphere(const glm::vec3& center, const float radius, const int matIndex);

	/**
	* @biref Calculates distance from ray origin to hit point. Calculations are done using sphere formula.
	* @param Ray which contains origin of ray (camera origin) and direction for individual pixel.
	* @return distance to hit point
	*/
	__device__ float hit(const Ray& ray) const;
	__device__ glm::vec3 getPosition() const { return position; };
	__device__ float getRadius() const { return radius; }
	__device__ int getMaterialIdx() const { return materialIndex; };

	void setMaterialIdx(int val) { materialIndex = val; };

	glm::vec3& getPositionRef() { return position; };
	float& getRadiusRef() { return radius; };
	int& getMaterialIdRef() { return materialIndex; };
	
private:
	float radius;
};
