#pragma once

#include "light.hpp"
#include <cstddef>

namespace onec
{
namespace uniform
{

struct Lighting
{
	static constexpr int maxPointLightCount{ 128 };
	static constexpr int maxSpotLightCount{ 128 };
	static constexpr int maxDirectionalLightCount{ 4 };
	
	PointLight pointLights[maxPointLightCount];
	SpotLight spotLights[maxSpotLightCount];
	DirectionalLight directionalLights[maxDirectionalLightCount];
	AmbientLight ambientLight;
	int pointLightCount{ 0 };
	int spotLightCount{ 0 };
	int directionalLightCount{ 0 };
};

}
}
