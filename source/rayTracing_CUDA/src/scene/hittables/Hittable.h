#pragma once
#include <glm/glm.hpp>
#include "Ray.h"

/**
* @brief Simple structure for storing material features.
*/
struct Material {
	glm::vec3 color;
	float roughness;
	float emission;

	/**
	* @brief Parametrized constructor with user specified values.
	*/
	Material(glm::vec3 color, float roughness, float emission): color(color), roughness(roughness), emission(emission){};

	/**
	* @brief Default constructor with init values.
	*/
	Material() : color(0.5f), roughness(0.f), emission(0.f){};

	/**
	* @brief Calculates power of material emission.
	* @return Color inlcuding power of emission
	* 
	*/
	__device__ glm::vec3 getEmissionPower() const {
		return color * emission;
	}
};

class Hittable {
public:
	const enum class Hit {
		NO_HIT = -1
	};
protected:
	Hittable(glm::vec3 position, int matIdx) : position(position), materialIndex(matIdx) {}
	glm::vec3 position;
	
	int materialIndex;
};
