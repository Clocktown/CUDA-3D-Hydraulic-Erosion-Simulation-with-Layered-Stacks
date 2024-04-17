#pragma once

#include <onec/resources/material.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace geo
{

struct PointRenderer
{
	std::shared_ptr<onec::Material> material;
	int first;
	int count;
};

}
