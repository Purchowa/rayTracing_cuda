#pragma once
#include <glm/glm.hpp>

/**
* @brief Simple strucutre for storing 3-coordinate source and direction vectors
*/
struct Ray
{
	glm::vec3 origin;
	glm::vec3 direction;
};