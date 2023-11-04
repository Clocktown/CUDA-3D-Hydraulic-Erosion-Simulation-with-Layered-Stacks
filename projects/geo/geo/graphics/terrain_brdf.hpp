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
		glm::vec4 waterColor{ 0.1f, 0.1f, 1.0f, 1.0f };
		glm::ivec2 gridSize{ 1024 };
		int maxLayerCount{ 8 };
		int pad;
	};

	Uniforms uniforms;
};

}
