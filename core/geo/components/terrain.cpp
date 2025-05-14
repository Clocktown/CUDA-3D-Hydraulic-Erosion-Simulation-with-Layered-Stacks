#include "terrain.hpp"
#include <onec/resources/graphics_buffer.hpp>
#include <glm/glm.hpp>

#include <onec/resources/mesh.hpp>

namespace geo
{

Terrain::Terrain(const glm::ivec2 gridSize, const float gridScale, const char maxLayerCount, const Simulation& simulation) :
	gridSize{ gridSize },
	gridScale{ gridScale },
	maxLayerCount{ maxLayerCount },
	simulation{ simulation }
{
	const int cellCount{ gridSize.x * gridSize.y };
	const int columnCount{ gridSize.x * gridSize.y * maxLayerCount };
	numValidColumns = columnCount;

	windowSize = glm::ivec2(1,1);
	screenTexture.initialize(GL_TEXTURE_2D, glm::ivec3(windowSize.x, windowSize.y, 0), GL_RGBA8, 1, onec::SamplerState{}, true);

	cudaTextureDesc texDesc{};
	texDesc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
	texDesc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
	texDesc.readMode = cudaTextureReadMode::cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	integratedBRDF.initialize(glm::ivec3(512, 512, 0), cudaCreateChannelDescHalf2(), 0, texDesc);
	layerCountBuffer.initialize(cellCount * static_cast<std::ptrdiff_t>(sizeof(char)), true);
	heightBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float4)), true);
	sedimentBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)), true);
	stabilityBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)), true);
	indicesBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(int)), true);
	pipeBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(char4)));
	slopeBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)));
	fluxBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float4)));
	slippageBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float4)));
	velocityBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float2)));
	damageBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)));
	sedimentFluxScaleBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float)));
	atomicCounter.initialize(static_cast<std::ptrdiff_t>(sizeof(int)));

	// Allocate QuadTree for Raymarching
	glm::ivec2 currentSize = gridSize;
	float currentScale = gridScale;
	int currentMaxLayerCount = maxLayerCount;
	for (int i = 0; i < NUM_QUADTREE_LAYERS; ++i) {
		currentSize = glm::ceil(0.5f * glm::vec2(currentSize));
		currentScale *= 2.f;
		if ((i % 2) == 0) {
			currentMaxLayerCount = glm::ceil(0.5f * currentMaxLayerCount);
		}

		const int cellCount{ currentSize.x * currentSize.y };
		const int columnCount{ cellCount * currentMaxLayerCount };

		quadTree[i].gridScale = currentScale;
		quadTree[i].gridSize = currentSize;
		quadTree[i].maxLayerCount = currentMaxLayerCount;
		quadTree[i].rGridScale = 1.f / currentScale;
		quadTree[i].layerStride = cellCount;
		quadTree[i].layerCountBuffer.initialize(cellCount * static_cast<std::ptrdiff_t>(sizeof(char)));
		quadTree[i].heightBuffer.initialize(columnCount * static_cast<std::ptrdiff_t>(sizeof(float4)));
	}
}

}
