#pragma once
#include <vector>
#include "hittables/Sphere.h"

/**
* @brief Simple structure for storing objects in scene
*/
struct Scene {
	std::vector<Sphere> sphere;
	std::vector<Material> material;
};