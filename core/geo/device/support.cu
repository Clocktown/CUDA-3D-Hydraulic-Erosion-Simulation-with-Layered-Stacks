#include "simulation.cuh"

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
			bool isTooClose = false;

			for (int layer{ layerCount - 1 }; layer >= 0; --layer, itFlatIndex -= simulation.layerStride)
			{
				auto heights = half4toVec4(simulation.heights[itFlatIndex]);
				float sediment = __half2float(simulation.sediments[itFlatIndex]);

				//heights[BEDROCK] += collapsedBedrock;
				if (isTooClose) {
					heights[BEDROCK] += heights[SAND] + collapsedBedrock;
					heights[SAND] = collapsedSand;
					heights[WATER] += collapsedWater;
					sediment += collapsedSediment;
				}
				else {
					const float sand = collapsedBedrock + collapsedSand;
					heights[SAND] += sand;
					heights[WATER] += collapsedWater;
					sediment += collapsedSediment;
				}
				if (previousCollapsed) {
					heights[CEILING] = previousCeiling;
				}
				collapsedBedrock = 0.f;
				collapsedSand = 0.f;
				collapsedWater = 0.f;
				collapsedSediment = 0.f;

				float stability = __half2float(simulation.stability[itFlatIndex]);
				const float floor = (layer == 0) ? -FLT_MAX : __half2float(((half*)(simulation.heights + itFlatIndex - simulation.layerStride))[CEILING]);
				const float bedrockBelow = (layer == 0) ? -FLT_MAX : __half2float(((half*)(simulation.heights + itFlatIndex - simulation.layerStride))[BEDROCK]);
				isTooClose = (layer > 0) && ((floor - bedrockBelow) < 0.05f * simulation.splitSize);
				const bool isTooThin = heights[BEDROCK] - floor <= simulation.minBedrockThickness;
				stability = (isTooThin || isTooClose) ? 0.f : stability;
				if (stability <= 0.f) {
					collapsedWater += heights[WATER];
					collapsedSand += heights[SAND];
					collapsedBedrock += heights[BEDROCK] - floor;
					// if a damaged column collapses, the damage has been converted to sand already but not yet been removed from the bedrock
					float damage = glm::min(__half2float(simulation.damages[itFlatIndex]), collapsedBedrock);
					collapsedBedrock -= damage;
					collapsedSediment += sediment;
					previousCeiling = heights[CEILING];
					heights = glm::vec4(0);
					sediment = 0.f;
					simulation.fluxes[itFlatIndex] = half4{{CUDART_ZERO_FP16, CUDART_ZERO_FP16}, {CUDART_ZERO_FP16, CUDART_ZERO_FP16}};
					previousCollapsed = true;
					newLayerCount--;
				}
				else {
					previousCollapsed = false;
				}

				simulation.heights[itFlatIndex] = toHalf4(heights);
				simulation.sediments[itFlatIndex] = __float2half(sediment);
				simulation.stability[itFlatIndex] = __float2half(stability);
			}

			itFlatIndex = flatIndex + simulation.layerStride; // set index to layer 1

			int tgtIndex = itFlatIndex;

			for (int layer{ 1 }; layer < layerCount; ++layer, itFlatIndex += simulation.layerStride)
			{
				if (simulation.stability[itFlatIndex] > CUDART_ZERO_FP16) {
					if (tgtIndex != itFlatIndex) {
						simulation.heights[tgtIndex] = simulation.heights[itFlatIndex];
						simulation.fluxes[tgtIndex] = simulation.fluxes[itFlatIndex];
						simulation.stability[tgtIndex] = simulation.stability[itFlatIndex];
						simulation.sediments[tgtIndex] = simulation.sediments[itFlatIndex];
						simulation.damages[tgtIndex] = simulation.damages[itFlatIndex];
					}
					tgtIndex += simulation.layerStride;
				}
			}

			simulation.layerCounts[flatIndex] = newLayerCount;
		}

		template<bool UseWeight>
		__global__ void stepSupportCheckKernel(int i)
		{
			glm::ivec2 index{ getLaunchIndex() };

			index.x <<= 1;
			index.x += (index.y + i) & 1;

			if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
			{
				return;
			}

			int flatIndex{ flattenIndex(index, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };

			float bedrockMin = __half2float(((half*)(simulation.heights + flatIndex))[CEILING]);
			flatIndex += simulation.layerStride; // Bottom layer is always stable, so skip

			for (int layer{ 1 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				glm::vec4 heights = half4toVec4(simulation.heights[flatIndex]);
				float bedrockMax = heights[BEDROCK];
				float oldStability = __half2float(simulation.stability[flatIndex]);
				oldStability = (oldStability == FLT_MAX) ? 0.f : oldStability; // If layer 0 is split in horizontal Erosion, layer 1 can have FLT_MAX as stability. Check might not really be necessary though
				float stability = 0.f;


				const float thickness = (bedrockMax - bedrockMin);
				const float iThickness = thickness > glm::epsilon<float>() ? 1.f / thickness : 0.f;
				float weight = fmaxf(thickness * simulation.bedrockDensity +
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

				float support = 0.f;

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
						auto heights = half4toVec4(simulation.heights[neighbor.flatIndex]);
						neighbor.stability = __half2float(simulation.stability[neighbor.flatIndex]);
						neighbor.bedrockMax = __half2float(((half*)(simulation.heights + neighbor.flatIndex))[BEDROCK]);

						float overlap = fminf(neighbor.bedrockMax, bedrockMax) - fmaxf(neighbor.bedrockMin, bedrockMin);
						if (overlap > 0.f) {
							float oSupport = simulation.gridScale * overlap * overlap * iThickness * simulation.bedrockSupport;
							support = fmaxf(fminf(oSupport, neighbor.stability), support);
						}

						neighbor.bedrockMin = __half2float(((half*)(simulation.heights + neighbor.flatIndex))[CEILING]);
					}
				}

				if constexpr (UseWeight) {
					stability = fmaxf(support - weight, oldStability);
				}
				else {
					stability = fmaxf(support, oldStability);
					stability = stability > 0.f ? 1.f : 0.f;
				}

				bedrockMin = heights[CEILING];
				simulation.stability[flatIndex] = __float2half(stability);
			}
		}

		void startSupportCheck(const Launch& launch) {
			CU_CHECK_KERNEL(startSupportCheckKernel << <launch.gridSize, launch.blockSize >> > ());
		}

		void endSupportCheck(const Launch& launch) {
			CU_CHECK_KERNEL(endSupportCheckKernel << <launch.gridSize, launch.blockSize >> > ());
		}

		void stepSupportCheck(const Launch& launch, bool use_weight) {
			auto newGrid = launch.gridSize;
			newGrid.x = (newGrid.x / 2) + 1;
			if (use_weight) {
				//CU_CHECK_KERNEL(stepSupportCheckKernel <true><< <launch.gridSize, launch.blockSize >> > ());
				CU_CHECK_KERNEL(stepSupportCheckKernel <true> << <newGrid, launch.blockSize >> > (0));
				CU_CHECK_KERNEL(stepSupportCheckKernel <true><< <newGrid, launch.blockSize >> > (1));
			}
			else {
				//CU_CHECK_KERNEL(stepSupportCheckKernel <false><< <launch.gridSize, launch.blockSize >> > ());
				CU_CHECK_KERNEL(stepSupportCheckKernel <false><< <newGrid, launch.blockSize >> > (0));
				CU_CHECK_KERNEL(stepSupportCheckKernel <false><< <newGrid, launch.blockSize >> > (1));
			}
		}
	}
}