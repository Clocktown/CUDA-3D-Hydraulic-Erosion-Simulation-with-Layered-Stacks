#pragma once

#include "light.hpp"

namespace onec
{
namespace device
{

struct Lighting
{
	static constexpr int maxPointLightCount{ 128 };
	static constexpr int maxSpotLightCount{ 128 };
	static constexpr int maxDirectionalLightCount{ 4 };
	
	PointLight pointLights[maxPointLightCount];
	SpotLight spotLights[maxSpotLightCount];
	DirectionalLight directionalLights[maxDirectionalLightCount];
	int pointLightCount{ 0 };
	int spotLightCount{ 0 };
	int directionalLightCount{ 0 };
};

}
}
