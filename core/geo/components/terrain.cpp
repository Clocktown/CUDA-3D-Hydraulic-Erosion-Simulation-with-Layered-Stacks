#include "terrain.hpp"
#include <onec/resources/graphics_buffer.hpp>
#include <glm/glm.hpp>

namespace geo
{

Terrain::Terrain(const glm::ivec2 gridSize, const float gridScale, const Simulation& simulation) :
	gridSize{ gridSize },
	gridScale{ gridScale },
	simulation{ simulation }
{
	const int cellCount{ gridSize.x * gridSize.y };
	const int columnCount{ gridSize.x * gridSize.y * maxLayerCount };

	layerCountBuffer.initialize(cellCount * static_cast<std::ptrdiff_t>(sizeof(char)), true);
	heightBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float4)), true);
	sedimentBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)), true);
	stabilityBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)), true);
	pipeBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(char4)));
	slopeBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)));
	fluxBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float4)));
}

}
