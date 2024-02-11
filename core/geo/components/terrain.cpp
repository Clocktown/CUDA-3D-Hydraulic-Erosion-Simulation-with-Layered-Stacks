#include "terrain.hpp"
#include <onec/resources/graphics_buffer.hpp>
#include <glm/glm.hpp>

namespace geo
{

Terrain::Terrain(const glm::ivec2 gridSize, const float gridScale, const Settings& settings) :
	gridSize{ gridSize },
	gridScale{ gridScale },
	settings{ settings }
{
	const int cellCount{ gridSize.x * gridSize.y };
	const int voxelCount{ gridSize.x * gridSize.y * maxLayerCount };

	layerCountBuffer.initialize(cellCount * static_cast<std::ptrdiff_t>(sizeof(int)), true);
	heightBuffer.initialize(voxelCount * static_cast<std::ptrdiff_t>(sizeof(float4)), true);
}

}
