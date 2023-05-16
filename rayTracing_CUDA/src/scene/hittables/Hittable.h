#pragma once
#include <glm/glm.hpp>
#include "Ray.h"

class Hittable {
public:
	const enum class Hit {
		NO_HIT = -1
	};
protected:
	Hittable(glm::vec3 position, glm::vec4 color) : position(position), color(color) {}
	glm::vec3 position;
	glm::vec4 color;
};