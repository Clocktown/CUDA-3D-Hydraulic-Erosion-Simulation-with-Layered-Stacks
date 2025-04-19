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

	return t.y >= t.x && t.y >= 0;
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
		const glm::vec3 inv_r_dir = 1.0f / r_dir;
		const auto inf_mask = glm::isinf(inv_r_dir);
		const glm::vec3 i_r_dir = glm::vec3(
			inf_mask.x ? 0.f : inv_r_dir.x,
			inf_mask.y ? 0.f : inv_r_dir.y,
			inf_mask.z ? 0.f : inv_r_dir.z
		);

		const glm::vec3 bmin = glm::vec3(-0.5f * simulation.gridScale * float(simulation.gridSize.x), -1000.f, -0.5f * simulation.gridScale * float(simulation.gridSize.y));
		const glm::vec3 bmax = glm::vec3(0.5f * simulation.gridScale * float(simulation.gridSize.x), 1000.f, 0.5f * simulation.gridScale * float(simulation.gridSize.y));
		glm::vec2 t;
		bool intersect = intersection(bmin, bmax, ro, i_r_dir, t);
		t.x = glm::max(t.x, 0.f);
		int steps = 0;
		bool hit = false;
		glm::vec3 normal = glm::vec3(0);
		if (intersect) {
			// DDA prep
			const glm::vec3 rayOriginGrid = glm::vec3(ro.x + t.x * r_dir.x - bmin.x, ro.y + t.x * r_dir.y, ro.z + t.x * r_dir.z - bmin.z);

			glm::ivec2 currentCell = glm::clamp(glm::ivec2(glm::floor(glm::vec2(rayOriginGrid.x, rayOriginGrid.z) * simulation.rGridScale)), glm::ivec2(0), simulation.gridSize - glm::ivec2(1));
			glm::ivec2 exit = glm::ivec2(r_dir.x < 0 ? -1 : simulation.gridSize.x, r_dir.z < 0 ? -1 : simulation.gridSize.y);
			glm::ivec2 step = glm::ivec2(r_dir.x < 0 ? -1 : 1, r_dir.z < 0 ? -1 : 1);

			glm::vec2 deltaT = glm::abs(glm::vec2(simulation.gridScale * i_r_dir.x, simulation.gridScale * i_r_dir.z));
			glm::vec2 T = t.x + ((glm::vec2(currentCell) + glm::clamp(1.f + glm::sign(glm::vec2(r_dir.x, r_dir.z)), 0.f, 1.f)) * simulation.gridScale - glm::vec2(rayOriginGrid.x, rayOriginGrid.z)) * glm::vec2(i_r_dir.x, i_r_dir.z);

			// DDA
			while (true) {
				steps++;
				// Test current Cell: TODO
				int flatIndex{ flattenIndex(currentCell, simulation.gridSize) };
				const int layerCount{ simulation.layerCounts[flatIndex] };
				//int itFlatIndex = flatIndex + (layerCount - 1) * simulation.layerStride;
				float floor = -FLT_MAX;

				float closest = FLT_MAX;
				const glm::vec2 bmin2D = glm::vec2(currentCell) * simulation.gridScale;
				const glm::vec2 bmax2D = bmin2D + simulation.gridScale;
				glm::vec2 boxHeights;

				for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
				{
					const auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
					const float totalTerrainHeight = terrainHeights[BEDROCK] + terrainHeights[SAND] + terrainHeights[WATER];

					glm::vec2 tempT;
					bool intersect = intersection(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), rayOriginGrid, i_r_dir, tempT);
					if (intersect && tempT.x < closest) {
						hit = true;
						boxHeights = glm::vec2(floor, totalTerrainHeight);
						closest = tempT.x;
					}

					floor = terrainHeights[CEILING];
				}
				if (hit) {
					glm::vec3 hitPos = rayOriginGrid + closest * r_dir;
					glm::vec3 bmin = glm::vec3(bmin2D.x, boxHeights.x, bmin2D.y);
					glm::vec3 bmax = glm::vec3(bmax2D.x, boxHeights.y, bmax2D.y);
					glm::vec3 diffMin = hitPos - bmin;
					glm::vec3 diffMax = hitPos - bmax;
					glm::vec3 distances = glm::min(glm::abs(diffMin), glm::abs(diffMax));

					// Determine the closest face based on the minimum distance
					if (distances.x <= distances.y && distances.x <= distances.z) {
						normal.x = (diffMin.x < 0.0f) ? -1.0f : 1.0f;
					} else if (distances.y <= distances.x && distances.y <= distances.z) {
						normal.y = (diffMin.y < 0.0f) ? -1.0f : 1.0f;
					} else {
						normal.z = (diffMin.z < 0.0f) ? -1.0f : 1.0f;
					}
					break;
				};
				// Advance to next cell
				const bool axis = T.x >= T.y;
				if (t.y < T[axis]) break;
				currentCell[axis] += step[axis];
				if (currentCell[axis] == exit[axis]) break;
				T[axis] += deltaT[axis];
			}
		}

		//normal = hit ? glm::vec3(1) : glm::vec3(-1);

		const glm::vec3 bgCol = glm::vec3(0);
		glm::vec3 col = 0.5f + 0.5f * normal;
		col = hit ? col : bgCol;



		col = glm::clamp(col, 0.f, 1.f);
		uchar4 val = uchar4(col.x * 255u, col.y * 255u, col.z * 255u, 255u);
		surf2Dwrite(val, simulation.screenSurface, index.x * 4, index.y, cudaBoundaryModeTrap);
}

void raymarchTerrain(const Launch& launch) {
	CU_CHECK_KERNEL(raymarchKernel << <launch.gridSize, launch.blockSize >> > ());
}
}
}