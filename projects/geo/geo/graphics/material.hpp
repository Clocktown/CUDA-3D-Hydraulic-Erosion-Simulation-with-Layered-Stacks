#pragma once

#include "terrain.hpp"
#include "../uniforms/material.hpp"
#include <onec/onec.hpp>

namespace geo
{

struct Material : public onec::Material
{
	explicit Material(Terrain& terrain);

	uniform::Material uniforms;
};

}
