#include "simulation.hpp"

namespace geo
{
	namespace device
	{
		__global__ void startSupportCheckKernel()
		{
			const glm::ivec2 index{ getLaunchIndex() };

			if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
			{
				return;
			}

			int flatIndex{ flattenIndex(index, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };

			for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				simulation.stability[flatIndex] = layer == 0.f ? FLT_MAX :  0.f;
				//simulation.stability[flatIndex] = float((layer + 1) % 2);
				//simulation.stability[flatIndex] = float(layer % 4 == 0);
			}
		}

		__global__ void endSupportCheckKernel()
		{
			const glm::ivec2 index{ getLaunchIndex() };

			if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
			{
				return;
			}

			const int flatIndex{ flattenIndex(index, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };
			int itFlatIndex = flatIndex + (layerCount - 1) * simulation.layerStride;

			int newLayerCount = layerCount;

			float collapsedWater = 0.f;
			float collapsedSand = 0.f;
			float collapsedSediment = 0.f;
			float collapsedBedrock = 0.f;
			float previousCeiling = FLT_MAX;

			bool previousCollapsed = false;

			for (int layer{ layerCount - 1 }; layer >= 0; --layer, itFlatIndex -= simulation.layerStride)
			{
				auto heights = glm::cuda_cast(simulation.heights[itFlatIndex]);
				float sediment = simulation.sediments[itFlatIndex];

				heights[BEDROCK] += collapsedBedrock;
				heights[SAND] += collapsedSand;
				heights[WATER] += collapsedWater;
				sediment += collapsedSediment;
				if (previousCollapsed) {
					heights[CEILING] = previousCeiling;
				}
				collapsedBedrock = 0.f;
				collapsedSand = 0.f;
				collapsedWater = 0.f;
				collapsedSediment = 0.f;

				float stability = simulation.stability[itFlatIndex];
				float floor = (layer == 0) ? -FLT_MAX : ((float*)(simulation.heights + itFlatIndex - simulation.layerStride))[CEILING];
				stability = (heights[BEDROCK] - floor <= simulation.minBedrockThickness) ? 0.f : stability;
				if (stability <= 0.f) {
					collapsedWater += heights[WATER];
					collapsedSand += heights[SAND];
					collapsedBedrock += heights[BEDROCK] - floor;
					collapsedSediment += sediment;
					previousCeiling = heights[CEILING];
					heights = glm::vec4(0);
					sediment = 0.f;
					simulation.fluxes[itFlatIndex] = float4(0.f);
					previousCollapsed = true;
					newLayerCount--;
				}
				else {
					previousCollapsed = false;
				}

				simulation.heights[itFlatIndex] = glm::cuda_cast(heights);
				simulation.sediments[itFlatIndex] = sediment;
			}

			itFlatIndex = flatIndex + simulation.layerStride; // set index to layer 1

			int tgtIndex = itFlatIndex;

			for (int layer{ 1 }; layer < layerCount; ++layer, itFlatIndex += simulation.layerStride)
			{
				float stability = simulation.stability[itFlatIndex];
				if (stability > 0.f) {
					if (tgtIndex != itFlatIndex) {
						simulation.heights[tgtIndex] = simulation.heights[itFlatIndex];
						simulation.fluxes[tgtIndex] = simulation.fluxes[itFlatIndex];
						simulation.stability[tgtIndex] = stability;
						simulation.sediments[tgtIndex] = simulation.sediments[itFlatIndex];
						simulation.damages[tgtIndex] = simulation.damages[itFlatIndex];
					}
					tgtIndex += simulation.layerStride;
				}
			}

			simulation.layerCounts[flatIndex] = newLayerCount;
		}

		template<bool UseWeight>
		__global__ void stepSupportCheckKernel()//int i)
		{
			const glm::ivec2 index{ getLaunchIndex() };

			if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
			{
				return;
			}

			//if ((index.x + ((index.y + i) % 2)) % 2 == 0) {
			//	return; // Checkerboard update to avoid race conditions (evil hack)
			//}

			int flatIndex{ flattenIndex(index, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };

			float bedrockMin = ((float*)(simulation.heights + flatIndex))[CEILING];
			flatIndex += simulation.layerStride; // Bottom layer is always stable, so skip

			for (int layer{ 1 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				glm::vec4 heights = glm::cuda_cast(simulation.heights[flatIndex]);
				float bedrockMax = heights[BEDROCK];
				float oldStability = simulation.stability[flatIndex];
				oldStability = (oldStability == FLT_MAX) ? 0.f : oldStability;
				float stability = 0.f;

				float weight = fmaxf((bedrockMax - bedrockMin) * simulation.bedrockDensity +
									heights[SAND] * simulation.sandDensity +
									heights[WATER] * simulation.waterDensity
					, 0.f);
				weight *= simulation.gridScale * simulation.gridScale;

				struct
				{
					glm::ivec2 index;
					int flatIndex;
					char layerCount;
					char layer;
					float bedrockMin;
					float bedrockMax;
					float stability;
				} neighbor;

				for (int i{ 0 }; i < 4; ++i)
				{
					neighbor.index = index + glm::cuda_cast(offsets[i]);

					if (isOutside(neighbor.index, simulation.gridSize))
					{
						stability += simulation.gridScale * fmaxf((bedrockMax - bedrockMin), 0.f) * simulation.borderSupport;
						continue;
					}

					neighbor.flatIndex = flattenIndex(neighbor.index, simulation.gridSize);
					neighbor.layerCount = simulation.layerCounts[neighbor.flatIndex];

					neighbor.bedrockMin = -FLT_MAX;

					for (neighbor.layer = 0; neighbor.layer < neighbor.layerCount; ++neighbor.layer, neighbor.flatIndex += simulation.layerStride)
					{
						auto heights = glm::cuda_cast(simulation.heights[neighbor.flatIndex]);
						neighbor.stability = simulation.stability[neighbor.flatIndex];
						neighbor.stability = (neighbor.layer > 0 && neighbor.stability == FLT_MAX) ? 0.f : neighbor.stability;
						neighbor.bedrockMax = ((float*)(simulation.heights + neighbor.flatIndex))[BEDROCK];

						float overlap = fminf(neighbor.bedrockMax, bedrockMax) - fmaxf(neighbor.bedrockMin, bedrockMin);
						if (overlap > 0.f && neighbor.stability > oldStability) {
							float support = simulation.gridScale * overlap * simulation.bedrockSupport;
							stability += fminf(support, neighbor.stability);
						}

						neighbor.bedrockMin = ((float*)(simulation.heights + neighbor.flatIndex))[CEILING];
					}
				}

				if constexpr (UseWeight) {
					stability = fmaxf(stability - weight, oldStability);
				}
				else {
					stability = fmaxf(stability, oldStability);
					stability = stability > 0.f ? 1.f : 0.f;
				}

				bedrockMin = heights[CEILING];
				simulation.stability[flatIndex] = stability;
			}
		}

		void startSupportCheck(const Launch& launch) {
			CU_CHECK_KERNEL(startSupportCheckKernel << <launch.gridSize, launch.blockSize >> > ());
		}

		void endSupportCheck(const Launch& launch) {
			CU_CHECK_KERNEL(endSupportCheckKernel << <launch.gridSize, launch.blockSize >> > ());
		}

		void stepSupportCheck(const Launch& launch, bool use_weight) {
			if (use_weight) {
				CU_CHECK_KERNEL(stepSupportCheckKernel <true><< <launch.gridSize, launch.blockSize >> > ());
				//CU_CHECK_KERNEL(stepSupportCheckKernel <true> << <launch.gridSize, launch.blockSize >> > (0));
				//CU_CHECK_KERNEL(stepSupportCheckKernel <true><< <launch.gridSize, launch.blockSize >> > (1));
			}
			else {
				CU_CHECK_KERNEL(stepSupportCheckKernel <false><< <launch.gridSize, launch.blockSize >> > ());
				//CU_CHECK_KERNEL(stepSupportCheckKernel <false><< <launch.gridSize, launch.blockSize >> > (0));
				//CU_CHECK_KERNEL(stepSupportCheckKernel <false><< <launch.gridSize, launch.blockSize >> > (1));
			}
		}
	}
}