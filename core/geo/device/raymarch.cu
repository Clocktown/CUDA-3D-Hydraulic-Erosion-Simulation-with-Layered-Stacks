#include "simulation.hpp"

namespace geo {
	namespace device {
__global__ void raymarchKernel() {
		const glm::ivec2 index{ getLaunchIndex() };

		if (isOutside(index, simulation.windowSize))
		{
			return;
		}
		
		uchar4 val = uchar4(255u, 0u, 0u, 255u);
		surf2Dwrite(val, simulation.screenSurface, index.x * 4, index.y, cudaBoundaryModeTrap);
}

void raymarchTerrain(const Launch& launch) {
	CU_CHECK_KERNEL(raymarchKernel << <launch.gridSize, launch.blockSize >> > ());
}
}
}