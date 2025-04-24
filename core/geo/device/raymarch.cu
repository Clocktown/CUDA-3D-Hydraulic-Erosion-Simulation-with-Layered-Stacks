#include "simulation.hpp"

#include <numbers>

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

__device__ __forceinline__ float intersectionVolume(const glm::vec3& b_min0, const glm::vec3& b_max0, const glm::vec3& b_min1, const glm::vec3& b_max1) {
	const glm::vec3 b_min = glm::max(b_min0, b_min1);
	const glm::vec3 b_max = glm::min(b_max0, b_max1);
	const glm::vec3 d = glm::max(b_max - b_min, glm::vec3(0));

	return d.x * d.y * d.z;
}

__global__ void raymarchDDAKernel() {
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

		const glm::vec3 bmin = glm::vec3(-0.5f * simulation.gridScale * float(simulation.gridSize.x), -FLT_MAX, -0.5f * simulation.gridScale * float(simulation.gridSize.y));
		const glm::vec3 bmax = glm::vec3(0.5f * simulation.gridScale * float(simulation.gridSize.x), FLT_MAX, 0.5f * simulation.gridScale * float(simulation.gridSize.y));
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
					// TODO: Test if a prior check improves performance (i.e. height-based)
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

__global__ void raymarchQuadTreeKernel() {
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

		const glm::vec3 bmin = glm::vec3(-0.5f * simulation.gridScale * float(simulation.gridSize.x), -FLT_MAX, -0.5f * simulation.gridScale * float(simulation.gridSize.y));
		const glm::vec3 bmax = glm::vec3(0.5f * simulation.gridScale * float(simulation.gridSize.x), FLT_MAX, 0.5f * simulation.gridScale * float(simulation.gridSize.y));
		glm::vec2 t;
		bool intersect = intersection(bmin, bmax, ro, i_r_dir, t);
		t.x = glm::max(t.x, 0.f);

		bool hit = false;
		glm::vec3 normal = glm::vec3(0);

		int currentLevel = geo::NUM_QUADTREE_LAYERS - 1;
		if (intersect) {
			// DDA prep
			glm::vec3 rayOriginGrid = glm::vec3(ro.x + t.x * r_dir.x - bmin.x, ro.y + t.x * r_dir.y, ro.z + t.x * r_dir.z - bmin.z);

			glm::ivec2 currentCell = glm::clamp(glm::ivec2(glm::floor(glm::vec2(rayOriginGrid.x, rayOriginGrid.z) * simulation.quadTree[currentLevel].rGridScale)), glm::ivec2(0), simulation.quadTree[currentLevel].gridSize - glm::ivec2(1));
			glm::ivec2 exit = glm::ivec2(r_dir.x < 0 ? -1 : simulation.quadTree[currentLevel].gridSize.x, r_dir.z < 0 ? -1 : simulation.quadTree[currentLevel].gridSize.y);
			glm::ivec2 step = glm::ivec2(r_dir.x < 0 ? -1 : 1, r_dir.z < 0 ? -1 : 1);

			glm::vec2 deltaT = glm::abs(glm::vec2(simulation.quadTree[currentLevel].gridScale * i_r_dir.x, simulation.quadTree[currentLevel].gridScale * i_r_dir.z));
			glm::vec2 T = t.x + ((glm::vec2(currentCell) + glm::clamp(1.f + glm::sign(glm::vec2(r_dir.x, r_dir.z)), 0.f, 1.f)) * simulation.quadTree[currentLevel].gridScale - glm::vec2(rayOriginGrid.x, rayOriginGrid.z)) * glm::vec2(i_r_dir.x, i_r_dir.z);

			// DDA
			while (true) {
				// Test current Cell: TODO
				glm::vec2 boxHeights;
				float closest = FLT_MAX;
				const glm::vec2 bmin2D = glm::vec2(currentCell) * simulation.gridScale;
				const glm::vec2 bmax2D = bmin2D + simulation.gridScale;

				if (currentLevel < 0) {
					// Actual Terrain
					int flatIndex{ flattenIndex(currentCell, simulation.gridSize) };
					const int layerCount{ simulation.layerCounts[flatIndex] };
					//int itFlatIndex = flatIndex + (layerCount - 1) * simulation.layerStride;
					float floor = -FLT_MAX;

					for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
					{
						const auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
						const float totalTerrainHeight = terrainHeights[BEDROCK] + terrainHeights[SAND] + terrainHeights[WATER];

						glm::vec2 tempT;
						// TODO: Test if a prior check improves performance (i.e. height-based)
						bool intersect = intersection(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), rayOriginGrid, i_r_dir, tempT);
						if (intersect && tempT.x < closest) {
							hit = true;
							boxHeights = glm::vec2(floor, totalTerrainHeight);
							closest = tempT.x;
						}

						floor = terrainHeights[CEILING];
					}
				}
				else {
					// QuadTree
					int flatIndex{ flattenIndex(currentCell, simulation.quadTree[currentLevel].gridSize) };
					const int layerCount{ simulation.quadTree[currentLevel].layerCounts[flatIndex] };
					//int itFlatIndex = flatIndex + (layerCount - 1) * simulation.layerStride;
					float floor = -FLT_MAX;

					const glm::vec2 bmin2D = glm::vec2(currentCell) * simulation.quadTree[currentLevel].gridScale;
					const glm::vec2 bmax2D = bmin2D + simulation.quadTree[currentLevel].gridScale;

					for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.quadTree[currentLevel].layerStride)
					{
						const auto terrainHeights = glm::cuda_cast(simulation.quadTree[currentLevel].heights[flatIndex]);
						const float totalTerrainHeight = terrainHeights[QFULLHEIGHT];

						glm::vec2 tempT;
						// TODO: Test if a prior check improves performance (i.e. height-based)
						bool intersect = intersection(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), rayOriginGrid, i_r_dir, tempT);
						if (intersect) {
							hit = true;
							t.x = tempT.x;
							break;
						}

						floor = terrainHeights[QCEILING];
					}
				}

				if (hit && (currentLevel < 0)) {
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
				}
				else if (hit) {
					rayOriginGrid = rayOriginGrid + t.x * r_dir;
					currentLevel--;
					if (currentLevel < 0) {
						currentCell = glm::clamp(glm::ivec2(glm::floor(glm::vec2(rayOriginGrid.x, rayOriginGrid.z) * simulation.rGridScale)), glm::ivec2(0), simulation.gridSize - glm::ivec2(1));
						exit = glm::ivec2(r_dir.x < 0 ? -1 : simulation.gridSize.x, r_dir.z < 0 ? -1 : simulation.gridSize.y);
						deltaT = glm::abs(glm::vec2(simulation.gridScale * i_r_dir.x, simulation.gridScale * i_r_dir.z));
						T = t.x + ((glm::vec2(currentCell) + glm::clamp(1.f + glm::sign(glm::vec2(r_dir.x, r_dir.z)), 0.f, 1.f)) * simulation.gridScale - glm::vec2(rayOriginGrid.x, rayOriginGrid.z)) * glm::vec2(i_r_dir.x, i_r_dir.z);
					}
					else {
						currentCell = glm::clamp(glm::ivec2(glm::floor(glm::vec2(rayOriginGrid.x, rayOriginGrid.z) * simulation.quadTree[currentLevel].rGridScale)), glm::ivec2(0), simulation.quadTree[currentLevel].gridSize - glm::ivec2(1));
						exit = glm::ivec2(r_dir.x < 0 ? -1 : simulation.quadTree[currentLevel].gridSize.x, r_dir.z < 0 ? -1 : simulation.quadTree[currentLevel].gridSize.y);
						deltaT = glm::abs(glm::vec2(simulation.quadTree[currentLevel].gridScale * i_r_dir.x, simulation.quadTree[currentLevel].gridScale * i_r_dir.z));
						T = t.x + ((glm::vec2(currentCell) + glm::clamp(1.f + glm::sign(glm::vec2(r_dir.x, r_dir.z)), 0.f, 1.f)) * simulation.quadTree[currentLevel].gridScale - glm::vec2(rayOriginGrid.x, rayOriginGrid.z)) * glm::vec2(i_r_dir.x, i_r_dir.z);
					}
					hit = false;
					continue;
				}
				// Advance to next cell
				const bool axis = T.x >= T.y;
				if (t.y < T[axis]) break;
				currentCell[axis] += step[axis];
				if (currentCell[axis] == exit[axis]) break;
				T[axis] += deltaT[axis];
				//t.x = T[axis];
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

__global__ void raymarchDDAQuadTestKernel(int level) {
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

		const glm::vec3 bmin = glm::vec3(-0.5f * simulation.quadTree[level].gridScale * float(simulation.quadTree[level].gridSize.x), -FLT_MAX, -0.5f * simulation.quadTree[level].gridScale * float(simulation.quadTree[level].gridSize.y));
		const glm::vec3 bmax = glm::vec3(0.5f * simulation.quadTree[level].gridScale * float(simulation.quadTree[level].gridSize.x), FLT_MAX, 0.5f * simulation.quadTree[level].gridScale * float(simulation.quadTree[level].gridSize.y));
		glm::vec2 t;
		bool intersect = intersection(bmin, bmax, ro, i_r_dir, t);
		t.x = glm::max(t.x, 0.f);
		int steps = 0;
		bool hit = false;
		glm::vec3 normal = glm::vec3(0);
		if (intersect) {
			// DDA prep
			const glm::vec3 rayOriginGrid = glm::vec3(ro.x + t.x * r_dir.x - bmin.x, ro.y + t.x * r_dir.y, ro.z + t.x * r_dir.z - bmin.z);

			glm::ivec2 currentCell = glm::clamp(glm::ivec2(glm::floor(glm::vec2(rayOriginGrid.x, rayOriginGrid.z) * simulation.quadTree[level].rGridScale)), glm::ivec2(0), simulation.quadTree[level].gridSize - glm::ivec2(1));
			glm::ivec2 exit = glm::ivec2(r_dir.x < 0 ? -1 : simulation.quadTree[level].gridSize.x, r_dir.z < 0 ? -1 : simulation.quadTree[level].gridSize.y);
			glm::ivec2 step = glm::ivec2(r_dir.x < 0 ? -1 : 1, r_dir.z < 0 ? -1 : 1);

			glm::vec2 deltaT = glm::abs(glm::vec2(simulation.quadTree[level].gridScale * i_r_dir.x, simulation.quadTree[level].gridScale * i_r_dir.z));
			glm::vec2 T = t.x + ((glm::vec2(currentCell) + glm::clamp(1.f + glm::sign(glm::vec2(r_dir.x, r_dir.z)), 0.f, 1.f)) * simulation.quadTree[level].gridScale - glm::vec2(rayOriginGrid.x, rayOriginGrid.z)) * glm::vec2(i_r_dir.x, i_r_dir.z);

			// DDA
			while (true) {
				steps++;
				int flatIndex{ flattenIndex(currentCell, simulation.quadTree[level].gridSize) };
				const int layerCount{ simulation.quadTree[level].layerCounts[flatIndex] };
				//int itFlatIndex = flatIndex + (layerCount - 1) * simulation.layerStride;
				float floor = -FLT_MAX;

				float closest = FLT_MAX;
				const glm::vec2 bmin2D = glm::vec2(currentCell) * simulation.quadTree[level].gridScale;
				const glm::vec2 bmax2D = bmin2D + simulation.quadTree[level].gridScale;
				glm::vec2 boxHeights;

				for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.quadTree[level].layerStride)
				{
					const auto terrainHeights = glm::cuda_cast(simulation.quadTree[level].heights[flatIndex]);
					const float totalTerrainHeight = terrainHeights[QFULLHEIGHT];//terrainHeights[BEDROCK] + terrainHeights[SAND] + terrainHeights[WATER];

					glm::vec2 tempT;
					// TODO: Test if a prior check improves performance (i.e. height-based)
					bool intersect = intersection(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), rayOriginGrid, i_r_dir, tempT);
					if (intersect && tempT.x < closest) {
						hit = true;
						boxHeights = glm::vec2(floor, totalTerrainHeight);
						closest = tempT.x;
					}

					floor = terrainHeights[QCEILING];// terrainHeights[CEILING];
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

__device__ __forceinline__ float getVolume(const glm::vec3& p, const float radius, const float radiusG) {
	const glm::vec3 boxBmin = p - radius;
	const glm::vec3 boxBmax = p + radius;

	const glm::vec2 pG = glm::vec2(p.x * simulation.rGridScale, p.z * simulation.rGridScale);

	float volume = 0.f;

	int y = glm::clamp(pG.y - radiusG, 0.f, float(simulation.gridSize.y));
	const float endY = glm::clamp(pG.y + radiusG, 0.f, float(simulation.gridSize.y));
	const float endX = glm::clamp(pG.x + radiusG, 0.f, float(simulation.gridSize.x));
	for (; y < endY; ++y) {
		int x = glm::clamp(pG.x - radiusG, 0.f, float(simulation.gridSize.x));
		for (; x < endX; ++x) {
			glm::ivec2 currentCell = glm::ivec2(x, y);

			//if (isOutside(currentCell, simulation.gridSize)) {
			//	continue;
			//}

			int flatIndex{ flattenIndex(currentCell, simulation.gridSize) };
			const int layerCount{ simulation.layerCounts[flatIndex] };
						
			float floor = -FLT_MAX;

			const glm::vec2 bmin2D = glm::vec2(currentCell) * simulation.gridScale;
			const glm::vec2 bmax2D = bmin2D + simulation.gridScale;

			for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
			{
				const auto terrainHeights = glm::cuda_cast(simulation.heights[flatIndex]);
				const float totalTerrainHeight = terrainHeights[BEDROCK] + terrainHeights[SAND] + terrainHeights[WATER];

				if (totalTerrainHeight <= boxBmin.y) {
					floor = terrainHeights[CEILING];
					continue;
				}

				volume += intersectionVolume(glm::vec3(bmin2D.x, floor, bmin2D.y), glm::vec3(bmax2D.x, totalTerrainHeight, bmax2D.y), boxBmin, boxBmax);

				floor = terrainHeights[CEILING];
				if (floor >= boxBmax.y) break;
			}
		}
	}
	return volume;
}

// How far can we step when computing the volume overlap of columns to an AABB of side length s? (see Arches paper definition of the implicit surface)
// My derivation:
// based on the last overlap, we calculate the volume change necessary to hit the surface (f(p) = 0)
// Moving along our ray, we retain part of the previous volume, remove some and add new volume.
// The fastest change would occur when moving diagonally through the AABB.
// Worst case, all the "old" volume was previously empty and all new volume is filled.
// The new volume gained is calculated as new_volume = s^3 - (s - distance/sqrt(3))^3.
// Solving for distance we get distance = 2*s*sqrt(3) - sqrt(3) (s + (s^3 - new_volume)^(1/3))
// However, for distances closer to s, the worst case changes to new_volume =  s^3 - (s - distance) * s * s 
// in which case distance = new_volume / (s * s)
// Using the necessary volume change from above as new_volume, we can calculate the safe distance.
// since f(p) = (2*i(p)/(s^3)) - 1, where i(p) is the volume overlap, given current volume X with f(p) > 0
// then the required volume Y is 0.5s^3, so the new volume gained is simply |X - (0.5s^3)|
__device__ __forceinline__ float calculateSafeStep(float volume, float targetVolume, float boxSize) {
	const float requiredVolumeGain = glm::abs(volume - targetVolume);
	const float boxSize2 = boxSize * boxSize;
	const float boxSize3 = boxSize2 * boxSize;
	const float d1 = requiredVolumeGain / boxSize2;
	constexpr float sqrt3 = std::numbers::sqrt3_v<float>;
	const float d2 = 2.f * boxSize * sqrt3 - sqrt3 * (boxSize + pow(boxSize3 - requiredVolumeGain, 1.f / 3.f));
	return glm::max(d1, d2);
}

__global__ void raymarchSmoothKernel(float volumePercentage, float radiusG, float normalSmoothingFactor) {
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

		// Danger: potential infinite loop for rays that never leave xz bounds
		const glm::vec3 bmin = glm::vec3(-0.5f * simulation.gridScale * float(simulation.gridSize.x), -FLT_MAX, -0.5f * simulation.gridScale * float(simulation.gridSize.y));
		const glm::vec3 bmax = glm::vec3(0.5f * simulation.gridScale * float(simulation.gridSize.x), FLT_MAX, 0.5f * simulation.gridScale * float(simulation.gridSize.y));
		glm::vec2 t;
		bool intersect = intersection(bmin, bmax, ro, i_r_dir, t);
		t.x = glm::max(t.x, 0.f);

		int steps = 0;
		bool hit = false;
		glm::vec3 p = glm::vec3(0);
		glm::vec3 n = glm::vec3(0);

		if (intersect) {
			// prep
			const glm::vec3 roGrid = glm::vec3(ro.x - bmin.x, ro.y, ro.z - bmin.z);

			const float radius = simulation.gridScale * radiusG;
			const float targetVolume = volumePercentage * 8.f * radius * radius * radius;

			// raymarching with box-filter for implicit surface
			while (t.x <= t.y) {
				steps++;
				p = roGrid + t.x * r_dir;
				const float volume = getVolume(p, radius, radiusG);

				if (volume > 0.99f * targetVolume) {
					hit = true;
					// TODO: keep normal using smoother terrain?
					const float e = normalSmoothingFactor * radius;
					const float eG = normalSmoothingFactor * radiusG;
					const float volumeN = normalSmoothingFactor != 1.f ? getVolume(p, e, eG) : volume;
					n = glm::normalize(glm::vec3(
						getVolume(p + glm::vec3(e, 0.f, 0.f), e, eG) - volumeN,
						getVolume(p + glm::vec3(0.f, e, 0.f), e, eG) - volumeN,
						getVolume(p + glm::vec3(0.f, 0.f, e), e, eG) - volumeN
					));
					break;
				}


				const float step = calculateSafeStep(volume, targetVolume, 2.f * radius);
				t.x += step;
			}
		}

		//normal = hit ? glm::vec3(1) : glm::vec3(-1);

		const glm::vec3 bgCol = glm::vec3(0);
		glm::vec3 col = 0.5f + 0.5f * n;
		col = hit ? col : bgCol;



		col = glm::clamp(col, 0.f, 1.f);
		uchar4 val = uchar4(col.x * 255u, col.y * 255u, col.z * 255u, 255u);
		surf2Dwrite(val, simulation.screenSurface, index.x * 4, index.y, cudaBoundaryModeTrap);
}

__global__ void buildQuadTreeFirstLayer() {
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.quadTree[0].gridSize))
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.quadTree[0].gridSize) };
	int maxOLayers = 0;

	struct Neighbor {
		glm::ivec2 index;
		int flatIndex;
		int layerCount;
		bool outside;
	} neighbors[4];

	for (int y = 0; y <= 1; ++y) {
		for (int x = 0; x <= 1; ++x) {
			const int oIndex = 2 * y + x;
			
			neighbors[oIndex].index = 2 * index + glm::ivec2(x, y);
			neighbors[oIndex].outside = isOutside(neighbors[oIndex].index, simulation.gridSize);
			if (neighbors[oIndex].outside)
			{
				continue;
			}

			neighbors[oIndex].flatIndex = flattenIndex(neighbors[oIndex].index, simulation.gridSize);
			neighbors[oIndex].layerCount = simulation.layerCounts[neighbors[oIndex].flatIndex];
			maxOLayers = glm::max(maxOLayers, neighbors[oIndex].layerCount);
		}
	}
	const char layerCount = glm::min(simulation.quadTree[0].maxLayerCount, maxOLayers);

	glm::vec4 heights;
	// Merge layer by layer, bottom to top
	for (int layer{ 0 }; layer < layerCount; ++layer) {
		heights = glm::vec4(-FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX);
		for (int y = 0; y <= 1; ++y) {
			for (int x = 0; x <= 1; ++x) {
				const int oIndex = 2 * y + x;

				if (neighbors[oIndex].outside || (neighbors[oIndex].layerCount < (layer + 1))) {
					continue;
				}
				const auto oTerrainHeights = glm::cuda_cast(simulation.heights[neighbors[oIndex].flatIndex + simulation.layerStride * layer]);
				// TODO: pad heights with smoothing radius? (splitting fullHeight to one padded up, one padded down as planned)
				const float solidHeight = oTerrainHeights[BEDROCK] + oTerrainHeights[SAND];
				const float fullHeight = solidHeight + oTerrainHeights[WATER];
				heights[QFULLHEIGHT] = glm::max(heights[QFULLHEIGHT], fullHeight);
				heights[QCEILING] = glm::min(heights[QCEILING], oTerrainHeights[CEILING]);
				heights[QSOLIDHEIGHT] = glm::max(heights[QSOLIDHEIGHT], solidHeight);
				heights[QAIR] = glm::min(heights[QAIR], fullHeight);
			}
		}
		simulation.quadTree[0].heights[flatIndex + simulation.quadTree[0].layerStride * layer] = glm::cuda_cast(heights);
	}

	simulation.quadTree[0].layerCounts[flatIndex] = layerCount;

	// Merge remaining columns into topmost column
	for (int layer{ simulation.quadTree[0].maxLayerCount - 1 }; layer < maxOLayers; ++layer) {
		for (int y = 0; y <= 1; ++y) {
			for (int x = 0; x <= 1; ++x) {
				const int oIndex = 2 * y + x;

				if (neighbors[oIndex].outside || (neighbors[oIndex].layerCount < (layer + 1))) {
					continue;
				}

				const auto oTerrainHeights = glm::cuda_cast(simulation.heights[neighbors[oIndex].flatIndex + simulation.layerStride * layer]);
				// TODO: pad heights with smoothing radius? (splitting fullHeight to one padded up, one padded down as planned)
				const float solidHeight = oTerrainHeights[BEDROCK] + oTerrainHeights[SAND];
				const float fullHeight = solidHeight + oTerrainHeights[WATER];
				heights[QFULLHEIGHT] = glm::max(heights[QFULLHEIGHT], fullHeight);
				heights[QCEILING] = glm::min(heights[QCEILING], oTerrainHeights[CEILING]);
				heights[QSOLIDHEIGHT] = glm::max(heights[QSOLIDHEIGHT], solidHeight);
				heights[QAIR] = glm::min(heights[QAIR], fullHeight);
			}
		}
	}

	if (maxOLayers > layerCount) {
		simulation.quadTree[0].heights[flatIndex + simulation.quadTree[0].layerStride * (layerCount - 1)] = glm::cuda_cast(heights);
	}
	// TODO: Compact tree (merge overlapping columns)
}

__global__ void buildQuadTreeLayer(int i) {
	const glm::ivec2 index{ getLaunchIndex() };

	if (isOutside(index, simulation.quadTree[i].gridSize))
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.quadTree[i].gridSize) };
	int maxOLayers = 0;

		struct Neighbor {
		glm::ivec2 index;
		int flatIndex;
		int layerCount;
		bool outside;
	} neighbors[4];

	for (int y = 0; y <= 1; ++y) {
		for (int x = 0; x <= 1; ++x) {
			const int oIndex = 2 * y + x;
			
			neighbors[oIndex].index = 2 * index + glm::ivec2(x, y);
			neighbors[oIndex].outside = isOutside(neighbors[oIndex].index, simulation.quadTree[i - 1].gridSize);
			if (neighbors[oIndex].outside)
			{
				continue;
			}

			neighbors[oIndex].flatIndex = flattenIndex(neighbors[oIndex].index, simulation.quadTree[i - 1].gridSize);
			neighbors[oIndex].layerCount = simulation.quadTree[i - 1].layerCounts[neighbors[oIndex].flatIndex];
			maxOLayers = glm::max(maxOLayers, neighbors[oIndex].layerCount);
		}
	}
	const char layerCount = glm::min(simulation.quadTree[i].maxLayerCount, maxOLayers);

	glm::vec4 heights;
	// Merge layer by layer, bottom to top
	for (int layer{ 0 }; layer < layerCount; ++layer) {
		heights = glm::vec4(-FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX);
		for (int y = 0; y <= 1; ++y) {
			for (int x = 0; x <= 1; ++x) {
				const int oIndex = 2 * y + x;

				if (neighbors[oIndex].outside || (neighbors[oIndex].layerCount < (layer + 1))) {
					continue;
				}

				const auto oTerrainHeights = glm::cuda_cast(simulation.quadTree[i - 1].heights[neighbors[oIndex].flatIndex + simulation.quadTree[i - 1].layerStride * layer]);
				// TODO: pad heights with smoothing radius? (splitting fullHeight to one padded up, one padded down as planned)
				heights[QFULLHEIGHT] = glm::max(heights[QFULLHEIGHT], oTerrainHeights[QFULLHEIGHT]);
				heights[QCEILING] = glm::min(heights[QCEILING], oTerrainHeights[QCEILING]);
				heights[QSOLIDHEIGHT] = glm::max(heights[QSOLIDHEIGHT], oTerrainHeights[QSOLIDHEIGHT]);
				heights[QAIR] = glm::min(heights[QAIR], oTerrainHeights[QAIR]);
			}
		}
		simulation.quadTree[i].heights[flatIndex + simulation.quadTree[i].layerStride * layer] = glm::cuda_cast(heights);
	}
	simulation.quadTree[i].layerCounts[flatIndex] = layerCount;

	// Merge remaining columns into topmost column
	for (int layer{ simulation.quadTree[i].maxLayerCount - 1 }; layer < maxOLayers; ++layer) {
		for (int y = 0; y <= 1; ++y) {
			for (int x = 0; x <= 1; ++x) {
				const int oIndex = 2 * y + x;

				if (neighbors[oIndex].outside || (neighbors[oIndex].layerCount < (layer + 1))) {
					continue;
				}

				const auto oTerrainHeights = glm::cuda_cast(simulation.quadTree[i - 1].heights[neighbors[oIndex].flatIndex + simulation.quadTree[i - 1].layerStride * layer]);
				// No Padding needed here, since base level has it already
				heights[QFULLHEIGHT] = glm::max(heights[QFULLHEIGHT], oTerrainHeights[QFULLHEIGHT]);
				heights[QCEILING] = glm::min(heights[QCEILING], oTerrainHeights[QCEILING]);
				heights[QSOLIDHEIGHT] = glm::max(heights[QSOLIDHEIGHT], oTerrainHeights[QSOLIDHEIGHT]);
				heights[QAIR] = glm::min(heights[QAIR], oTerrainHeights[QAIR]);
			}
		}
	}

	if (maxOLayers > layerCount) {
		simulation.quadTree[i].heights[flatIndex + simulation.quadTree[i].layerStride * (layerCount - 1)] = glm::cuda_cast(heights);
	}
	// TODO: Compact tree (merge overlapping columns)
}

void buildQuadTree(const std::vector<Launch>& launch) {
	// first layer (special)
	CU_CHECK_KERNEL(buildQuadTreeFirstLayer << <launch[0].gridSize, launch[0].blockSize >> > ());
	// remaining layers
	for (int i = 1; i < launch.size(); ++i) {
		CU_CHECK_KERNEL(buildQuadTreeLayer << <launch[i].gridSize, launch[i].blockSize >> > (i));
	}
}

void raymarchTerrain(const Launch& launch, bool useInterpolation, float volumePercentage, float smoothingRadiusInCells, float normalSmoothingFactor, int debugLayer) {
	if (useInterpolation) {
		CU_CHECK_KERNEL(raymarchSmoothKernel << <launch.gridSize, launch.blockSize >> > (volumePercentage, smoothingRadiusInCells, normalSmoothingFactor));
	}
	else {
		if (debugLayer < 0) {
			CU_CHECK_KERNEL(raymarchQuadTreeKernel << <launch.gridSize, launch.blockSize >> > ());
		}
		else {
			CU_CHECK_KERNEL(raymarchDDAQuadTestKernel << <launch.gridSize, launch.blockSize >> > (debugLayer));
		}
	}
}
}
}