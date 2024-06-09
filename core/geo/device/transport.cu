#include "simulation.hpp"

namespace geo
{
namespace device
{

template <bool outflowBorders, bool slippageOutflowBorders>
__global__ void pipeKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	const float gridScale2{ 2.0f * simulation.gridScale };

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		const glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		const float sand{ height[BEDROCK] + height[SAND] };
		const float water{ sand + height[WATER] };

		glm::vec<4, char> pipe{ -1 };
		glm::vec4 heights{ sand };
		glm::vec4 flux{ glm::cuda_cast(simulation.fluxes[flatIndex]) };
		glm::vec4 slippage{ 0.0f };

		struct
		{
			glm::ivec2 index;
			int flatIndex;
			int layerCount;
			int layer;
			glm::vec4 height;
			float sand;
			float water;
		} neighbor;

		for (int i{ 0 }; i < 4; ++i)
		{
			neighbor.index = index + glm::cuda_cast(offsets[i]);

			if (isOutside(neighbor.index, simulation.gridSize))
			{
				if constexpr (outflowBorders) {
					heights[i] = height[BEDROCK];

					const float deltaHeight{ water - height[BEDROCK]};
					const float crossSectionalArea{ simulation.gridScale * simulation.gridScale }; 

					flux[i] = glm::max(((1.f - 0.01f * simulation.deltaTime) * flux[i]) - simulation.deltaTime * crossSectionalArea * simulation.gravity * deltaHeight * simulation.rGridScale, 0.0f);
				}
				if constexpr (slippageOutflowBorders) {
					const float avgWater{ 0.5f * height[WATER] };
					const float talusSlope{ glm::mix(simulation.dryTalusSlope, simulation.wetTalusSlope, glm::min(avgWater * simulation.iSlippageInterpolationRange, 1.0f)) };

					slippage[i] = glm::max(
							glm::max(0.125f * (sand - height[BEDROCK] - talusSlope * simulation.gridScale), 0.0f)
						, 0.f);
				}
				continue;
			}

			neighbor.flatIndex = flattenIndex(neighbor.index, simulation.gridSize);
			neighbor.layerCount = simulation.layerCounts[neighbor.flatIndex];

			for (neighbor.layer = 0; neighbor.layer < neighbor.layerCount; ++neighbor.layer, neighbor.flatIndex += simulation.layerStride)
			{
				neighbor.height = glm::cuda_cast(simulation.heights[neighbor.flatIndex]);
				neighbor.sand = neighbor.height[BEDROCK] + neighbor.height[SAND];
				neighbor.water = neighbor.sand + neighbor.height[WATER];

				if (height[BEDROCK] < neighbor.height[CEILING] && (neighbor.water + glm::epsilon<float>()) < neighbor.height[CEILING])
				{
					const float deltaHeight{ glm::min(water, neighbor.height[CEILING]) - glm::max(neighbor.water, sand)  };
					const float crossSectionalArea{ simulation.gridScale * simulation.gridScale }; // dynamic?

					pipe[i] = static_cast<char>(neighbor.layer);
					heights[i] = neighbor.sand;
					flux[i] = glm::max(((1.f - 0.01f * simulation.deltaTime) * flux[i]) - simulation.deltaTime * crossSectionalArea * simulation.gravity * deltaHeight * simulation.rGridScale, 0.0f);

					if (neighbor.height[CEILING] < FLT_MAX)
					{
						const float freeSpace{ glm::max((neighbor.height[CEILING] - neighbor.water), 0.f) };
						const float takenSpace{ flux[i] * simulation.rGridScale * simulation.rGridScale * simulation.deltaTime };

						flux[i] *= glm::min(freeSpace / (takenSpace + glm::epsilon<float>()), 1.0f);
					}

					const float avgWater{ 0.5f * (height[WATER] + neighbor.height[WATER]) };
					const float talusSlope{ glm::mix(simulation.dryTalusSlope, simulation.wetTalusSlope, glm::min(avgWater * simulation.iSlippageInterpolationRange, 1.0f)) };

					slippage[i] = glm::max(
						glm::min(
							glm::max(0.125f * (sand - neighbor.sand - talusSlope * simulation.gridScale), 0.0f), 
							0.25f * (neighbor.height[CEILING] - neighbor.height[BEDROCK] - neighbor.height[WATER] - neighbor.height[SAND])
						)
						, 0.f);

					break;
				}
			}
		}

		const glm::vec3 tangents[2]{ glm::normalize(glm::vec3{ gridScale2, heights[RIGHT] - heights[LEFT], 0.0f }),
									 glm::normalize(glm::vec3{ 0.0f, heights[UP] - heights[DOWN], gridScale2 }) };

		const glm::vec3 normal{ glm::cross(tangents[0], tangents[1]) };
		const float slope{ glm::sqrt(1.0f - normal.y * normal.y) }; // sin(alpha)

		flux *= glm::min(height[WATER] * simulation.gridScale * simulation.gridScale /
						 ((flux.x + flux.y + flux.z + flux.w) * simulation.deltaTime + glm::epsilon<float>()), 1.0f);
		const float sedimentFluxScale{ glm::min(simulation.sediments[flatIndex] * simulation.gridScale * simulation.gridScale /
						 ((flux.x + flux.y + flux.z + flux.w) * simulation.deltaTime + glm::epsilon<float>()), 1.0f)};
		simulation.sedimentFluxScale[flatIndex] = sedimentFluxScale;

		slippage *= glm::min(height[SAND] / (slippage.x + slippage.y + slippage.z + slippage.w + glm::epsilon<float>()), 1.0f);

		simulation.pipes[flatIndex] = glm::cuda_cast(pipe);
		simulation.slopes[flatIndex] = slope;
		simulation.fluxes[flatIndex] = glm::cuda_cast(flux);
		simulation.slippages[flatIndex] = glm::cuda_cast(slippage);
	}
}

template <bool enableSlippage>
__global__ void transportKernel()
{
	const glm::ivec2 index{ getLaunchIndex() };

	if (index.x >= simulation.gridSize.x || index.y >= simulation.gridSize.y)
	{
		return;
	}

	int flatIndex{ flattenIndex(index, simulation.gridSize) };
	const int layerCount{ simulation.layerCounts[flatIndex] };

	const float integrationScale{ simulation.rGridScale * simulation.rGridScale * simulation.deltaTime };

	for (int layer{ 0 }; layer < layerCount; ++layer, flatIndex += simulation.layerStride)
	{
		glm::vec4 height{ glm::cuda_cast(simulation.heights[flatIndex]) };
		float sediment{ simulation.sediments[flatIndex] };
		glm::vec4 flux{ glm::cuda_cast(simulation.fluxes[flatIndex]) };
		glm::vec4 sedimentFlux{ simulation.sedimentFluxScale[flatIndex] * flux };
		glm::vec4 slippage{ glm::cuda_cast(simulation.slippages[flatIndex]) };

		struct
		{
			glm::ivec2 index;
			int flatIndex;
			int flatIndex4;
			int layerCount;
			int layer;
			float sediment;
			float sedimentFluxScale;
			float flux;
			float slippage;
		} neighbor;

		for (int i{ 0 }; i < 4; ++i)
		{
			neighbor.index = index + glm::cuda_cast(offsets[i]);

			if (isOutside(neighbor.index, simulation.gridSize))
			{
				continue;
			}

			neighbor.flatIndex = flattenIndex(neighbor.index, simulation.gridSize);
			neighbor.layerCount = simulation.layerCounts[neighbor.flatIndex];
			
			const int direction{ (i + 2) % 4 };

			for (neighbor.layer = 0; neighbor.layer < neighbor.layerCount; ++neighbor.layer, neighbor.flatIndex += simulation.layerStride)
			{
				neighbor.flatIndex4 = 4 * neighbor.flatIndex;

				if (reinterpret_cast<char*>(simulation.pipes)[neighbor.flatIndex4 + direction] == layer)
				{
					neighbor.sediment = simulation.sediments[neighbor.flatIndex];
					neighbor.sedimentFluxScale = simulation.sedimentFluxScale[neighbor.flatIndex];
					neighbor.flux = reinterpret_cast<float*>(simulation.fluxes)[neighbor.flatIndex4 + direction];
					neighbor.slippage = reinterpret_cast<float*>(simulation.slippages)[neighbor.flatIndex4 + direction];

					flux[i] -= neighbor.flux;
					sedimentFlux[i] -= neighbor.sedimentFluxScale * neighbor.flux;
					slippage[i] -= neighbor.slippage;
				}
			}
		}

		float avgWater{ height[WATER] };

		height[WATER] = glm::clamp(height[WATER] - integrationScale * (flux.x + flux.y + flux.z + flux.w), 0.0f, glm::max(height[CEILING] - height[BEDROCK] - height[SAND], 0.f));
		const float evaporationScale = glm::clamp(simulation.iEvaporationEmptySpaceScale * (height[CEILING] - height[BEDROCK] - height[SAND] - height[WATER]), 0.01f, 1.0f);
		height[WATER] = glm::max(height[WATER] - simulation.evaporation * evaporationScale * simulation.deltaTime, 0.0f);
		// avgWater = 0.5f * (avgWater + height[WATER]);
		const glm::vec2 velocity{ simulation.rGridScale * simulation.rGridScale * 0.5f * glm::vec2(flux[RIGHT] - flux[LEFT], flux[UP] - flux[DOWN]) };

		//const float mag = glm::length(velocity);
		//if (layer == (layerCount - 1) && index.x ==  int(glm::ceil(70.f/256.f)) && index.y == 0) {
		//	printf("%f\n", mag);
		//}

		if constexpr (enableSlippage) {
			height[SAND] = glm::clamp(height[SAND] - (slippage.x + slippage.y + slippage.z + slippage.w), 0.0f, height[CEILING] - height[BEDROCK] - height[WATER]);
		}

		const float petrificationAmount{ glm::min(simulation.petrification * simulation.deltaTime * height[SAND], height[SAND]) };
		height[BEDROCK] += petrificationAmount;
		height[SAND] -= petrificationAmount;

		sediment = glm::max(sediment - integrationScale * (sedimentFlux.x + sedimentFlux.y + sedimentFlux.z + sedimentFlux.w), 0.0f);

		simulation.heights[flatIndex] = glm::cuda_cast(height);
		simulation.sediments[flatIndex] = sediment;
		simulation.velocities[flatIndex] = glm::cuda_cast(velocity);
	}
}

void transport(const Launch& launch, bool enable_slippage, bool use_outflow_borders, bool use_slippage_outflow_borders, geo::performance& perf)
{
	if (perf.measureIndividualKernels) perf.measurements["Setup Pipes"].start();
	if (use_outflow_borders) {
		if (use_slippage_outflow_borders) {
			CU_CHECK_KERNEL(pipeKernel<true, true><<<launch.gridSize, launch.blockSize>>>());

		}
		else {
			CU_CHECK_KERNEL(pipeKernel<true,false><<<launch.gridSize, launch.blockSize>>>());
		}
	}
	else {
		if (use_slippage_outflow_borders) {
			CU_CHECK_KERNEL(pipeKernel<false, true><<<launch.gridSize, launch.blockSize>>>());

		}
		else {
			CU_CHECK_KERNEL(pipeKernel<false,false><<<launch.gridSize, launch.blockSize>>>());
		}
	}
	if (perf.measureIndividualKernels) perf.measurements["Setup Pipes"].stop();

	if (perf.measureIndividualKernels) perf.measurements["Resolve Pipes"].start();
	if (enable_slippage) {
		CU_CHECK_KERNEL(transportKernel<true><<<launch.gridSize, launch.blockSize>>>());
	}
	else {
		CU_CHECK_KERNEL(transportKernel<false><<<launch.gridSize, launch.blockSize>>>());
	}
	if (perf.measureIndividualKernels) perf.measurements["Resolve Pipes"].stop();
}

}
}
