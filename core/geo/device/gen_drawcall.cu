#include "simulation.hpp"

namespace geo
{
	namespace device
	{

		__global__ void generateDrawCallsKernel()
		{
			const glm::ivec2 index{ getLaunchIndex() };

			if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
			{
				return;
			}

			int flatIndex{ flattenIndex(index, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };

			DrawElementsIndirectCommand cmd;
			cmd.count = 36;
			cmd.baseVertex = 0;
			cmd.instanceCount = layerCount;
			cmd.firstIndex = 0;
			cmd.baseInstance = 0;

			simulation.drawCalls[flatIndex] = cmd;
		}

		void generateDrawCalls(const Launch& launch)
		{
			CU_CHECK_KERNEL(generateDrawCallsKernel << <launch.gridSize, launch.blockSize >> > ());
		}

	}
}
