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
			const char layerCount{ simulation.layerCounts[flatIndex] };

			for (char layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				simulation.stability[flatIndex] = layer == 0.f ? FLT_MAX :  0.f;
			}
		}

		__global__ void stepSupportCheckKernel()
		{
			const glm::ivec2 index{ getLaunchIndex() };

			if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
			{
				return;
			}

			int flatIndex{ flattenIndex(index, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };

			float bedrockMin = ((float*)(simulation.heights + flatIndex))[CEILING];
			flatIndex += simulation.layerStride; // Bottom layer is always stable, so skip

			for (int layer{ 1 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				float bedrockMax = ((float*)(simulation.heights + flatIndex))[BEDROCK];
				float oldStability = simulation.stability[flatIndex];
				float stability = 0.f;

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
						stability += simulation.borderSupport;
						continue;
					}

					neighbor.flatIndex = flattenIndex(neighbor.index, simulation.gridSize);
					neighbor.layerCount = simulation.layerCounts[neighbor.flatIndex];

					neighbor.bedrockMin = -FLT_MAX;

					for (neighbor.layer = 0; neighbor.layer < neighbor.layerCount; ++neighbor.layer, neighbor.flatIndex += simulation.layerStride)
					{
						auto heights = glm::cuda_cast(simulation.heights[neighbor.flatIndex]);
						neighbor.stability = simulation.stability[neighbor.flatIndex];
						neighbor.bedrockMax = ((float*)(simulation.heights + neighbor.flatIndex))[BEDROCK];;

						// TODO: Overlap Check, compute stability

						neighbor.bedrockMin = ((float*)(simulation.heights + neighbor.flatIndex))[CEILING];;
					}
				}

				bedrockMin = ((float*)(simulation.heights + flatIndex))[CEILING];
				simulation.stability[flatIndex] = stability;
			}
		}

		void startSupportCheck(const Launch& launch) {
			CU_CHECK_KERNEL(startSupportCheckKernel << <launch.gridSize, launch.blockSize >> > ());
		}

		void endSupportCheck(const Launch& launch) {

		}

		void stepSupportCheck(const Launch& launch) {
			CU_CHECK_KERNEL(stepSupportCheckKernel << <launch.gridSize, launch.blockSize >> > ());
		}
	}
}