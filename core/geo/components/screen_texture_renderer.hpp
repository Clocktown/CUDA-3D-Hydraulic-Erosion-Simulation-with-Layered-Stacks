#pragma once

#include <onec/resources/material.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace geo
{

struct ScreenTextureRenderer
{
	std::shared_ptr<onec::Material> material;
	bool enabled;
};

}
