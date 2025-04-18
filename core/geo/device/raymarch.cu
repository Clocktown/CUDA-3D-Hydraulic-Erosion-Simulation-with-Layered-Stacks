#include "simulation.hpp"

namespace geo {
	namespace device {

__device__ __forceinline__ bool intersection(const glm::vec3& b_min, const glm::vec3& b_max, const glm::vec3& r_o, const glm::vec3& inv_r_dir, glm::vec2& t) {
    glm::vec3 t1 = (b_min - r_o) * inv_r_dir;
    glm::vec3 t2 = (b_max - r_o) * inv_r_dir;

    glm::vec3 vtmin = glm::min(t1, t2);
    glm::vec3 vtmax = glm::max(t1, t2);

    t.x = glm::max(vtmin.x, glm::max(vtmin.y, vtmin.z));
    t.y = glm::min(vtmax.x, glm::min(vtmax.y, vtmax.z));

    return t.y >= t.x && t.x >= 0;
}

__global__ void raymarchKernel() {
		const glm::ivec2 index{ getLaunchIndex() };

		if (isOutside(index, simulation.windowSize))
		{
			return;
		}

		const glm::vec3 ro = simulation.camPos;
		const glm::vec3 pW = simulation.lowerLeft + (index.x + 0.5f) * simulation.rightVec + (index.y + 0.5f) * simulation.upVec;
		const glm::vec3 r_dir = glm::normalize(pW - ro);
		const glm::vec3 bmin = glm::vec3(-0.5f * simulation.gridScale * float(simulation.gridSize.x), -1000.f, -0.5f * simulation.gridScale * float(simulation.gridSize.y));
		const glm::vec3 bmax = glm::vec3(0.5f * simulation.gridScale * float(simulation.gridSize.x), 1000.f, 0.5f * simulation.gridScale * float(simulation.gridSize.y));
		glm::vec2 t;
		bool intersect = intersection(bmin, bmax, ro, 1.0f / r_dir, t);
		
		uchar4 val = uchar4(intersect ? 0u : 255u, 0u, 0u, 255u);
		surf2Dwrite(val, simulation.screenSurface, index.x * 4, index.y, cudaBoundaryModeTrap);
}

void raymarchTerrain(const Launch& launch) {
	CU_CHECK_KERNEL(raymarchKernel << <launch.gridSize, launch.blockSize >> > ());
}
}
}