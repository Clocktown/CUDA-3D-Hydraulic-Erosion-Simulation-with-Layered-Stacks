#pragma once

#include "../device/terrain_brdf.hpp"
#include <onec/onec.hpp>

namespace geo
{

struct TerrainBRDF : public onec::Material
{
	device::TerrainBRDF uniforms;
};

}
