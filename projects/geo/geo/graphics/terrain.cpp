#include "terrain.hpp"

namespace geo
{

Terrain::Terrain(const glm::ivec3 gridSize, const float gridScale) :
	gridSize{ gridSize },
	gridScale{ gridScale }
{
	const onec::SamplerState samplerState;
	infoMap.initialize(GL_TEXTURE_3D, gridSize, GL_RGBA8I, 1, samplerState, true, false, true);
	heightMap.initialize(GL_TEXTURE_3D, gridSize, GL_RGBA32F, 1, samplerState, true, false, true);
}

}
