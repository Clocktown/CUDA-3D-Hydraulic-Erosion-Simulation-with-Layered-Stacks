#pragma once

#include <onec/onec.hpp>

namespace geo
{

struct TerrainBRDF : public onec::Material
{
	struct Uniforms
	{
		glm::vec4 bedrockColor{ 0.5f, 0.5f, 0.5f, 1.0f };
		glm::vec4 sandColor{ 0.9f, 0.8f, 0.6f, 1.0f };
		glm::vec4 waterColor{ 0.0f, 0.3f, 0.75f, 1.0f };
		glm::ivec2 gridSize{ 1024 };
		float gridScale{ 1.0f };
		int maxLayerCount{ 8 };
	};

	Uniforms uniforms;
};

}
