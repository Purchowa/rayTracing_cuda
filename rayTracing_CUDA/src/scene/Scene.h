#pragma once
#include <vector>
#include "hittables/Sphere.h"

struct Scene {
	std::vector<Sphere> sphere;
};