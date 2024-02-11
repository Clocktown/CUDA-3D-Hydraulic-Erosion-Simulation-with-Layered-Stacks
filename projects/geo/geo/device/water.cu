#include "kernels.hpp"
#include "common.hpp"

namespace geo
{
namespace device
{

__global__ void rainKernel(Simulation simulation)
{
	glm::ivec3 index{ glm::ivec2{ getLaunchIndex() }, 0 };

	if (isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	glm::int8 z;

	do
	{
		z = index.z;
		index.z = simulation.infoArray.read<glm::i8vec4>(index)[ABOVE];
	}
	while (index.z != INVALID_INDEX);

	index.z = z;

	glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };
	height[WATER] += simulation.rain * simulation.gridScale * simulation.gridScale * simulation.deltaTime;

	simulation.heightArray.write(index, height);
}

__global__ void fluxKernel(Simulation simulation)
{
	glm::ivec3 index{ glm::ivec2{ getLaunchIndex() }, 0 };

	if (isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	do
	{
		const glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };
		const float solidHeight{ height[BEDROCK] + height[SAND] };
		const float totalHeight{ solidHeight + height[WATER] };

		glm::i8vec4 pipes;
		glm::vec4 flux{ simulation.fluxArray.read<glm::vec4>(index) };

		struct Neighbor
		{
			glm::ivec3 index;
			glm::vec4 height;
			float solidHeight;
			float totalHeight;
		};

		const Neighborhood neighborhood;
		
		for (int i{ 0 }; i < neighborhood.count; ++i)
		{
			Neighbor neighbor;
			neighbor.index.x = index.x + neighborhood.offsets[i].x;
			neighbor.index.y = index.y + neighborhood.offsets[i].y;

			if (isOutside(glm::ivec2{ neighbor.index }, glm::ivec2{ simulation.gridSize }))
			{
				pipes[i] = INVALID_INDEX;
				flux[i] = 0.0f;

				continue;
			}

			neighbor.index.z = 0;

			do
			{
				neighbor.height = simulation.heightArray.read<glm::vec4>(neighbor.index);
				neighbor.solidHeight = neighbor.height[BEDROCK] + neighbor.height[SAND];
				neighbor.totalHeight = neighbor.solidHeight + neighbor.height[WATER];

				if (solidHeight < neighbor.height[MAX_HEIGHT] && neighbor.height[MAX_HEIGHT] - neighbor.totalHeight > 1e-6f)
				{
					const float heightDifference{ totalHeight - neighbor.totalHeight };
					//const float heightDifference{ totalHeight - glm::max(neighbor.totalHeight, solidHeight) }; // 2013 Interaction with Dynamic Large Bodies in Efficient, Real-Time Water Simulation
					
					pipes[i] = neighbor.index.z;
					flux[i] = (heightDifference > 0.0f) *
						      glm::max(flux[i] - heightDifference * simulation.gravity * simulation.gridScale * simulation.deltaTime, 0.0f);
					
					if (neighbor.height[MAX_HEIGHT] < FLT_MAX)
					{
						const float freeSpace{ neighbor.height[MAX_HEIGHT] - neighbor.totalHeight * simulation.gridScale * simulation.gridScale };
						const float takenSpace{ flux[i] * simulation.deltaTime };

						flux[i] *= glm::min(freeSpace / (takenSpace + glm::epsilon<float>()), 1.0f);
					}

					break;
				}
				else if (totalHeight < neighbor.height[MAX_HEIGHT])
				{
					break;
				}

				neighbor.index.z = simulation.infoArray.read<glm::i8vec4>(neighbor.index)[ABOVE];

				if (neighbor.index.z == INVALID_INDEX)
				{
					pipes[i] = INVALID_INDEX;
					flux[i] = 0.0f;

					break;
				}
			}
			while (true);
		}

		flux *= glm::min(height[WATER] * simulation.gridScale * simulation.gridScale / 
						 ((flux[RIGHT] + flux[UP] + flux[LEFT] + flux[DOWN] + glm::epsilon<float>()) * simulation.deltaTime), 1.0f);
		
		simulation.pipeArray.write(index, pipes);
		simulation.fluxArray.write(index, flux);

		const float fluxStrength{ glm::length(flux) };

		if (fluxStrength > 0.0f)
		{
			const float sediment{ simulation.flowArray.read<glm::vec4>(index)[SEDIMENT] };
			const glm::vec4 sedimentFlux{ sediment * flux / fluxStrength };

			simulation.sedimentFluxArray.write(index, sedimentFlux);
		}

		index.z = simulation.infoArray.read<glm::i8vec4>(index)[ABOVE];
	}
	while (index.z != INVALID_INDEX);
}

__global__ void waterKernel(Simulation simulation)
{
	glm::ivec3 index{ glm::ivec2{ getLaunchIndex() }, 0 };

	if (isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	do
	{
		glm::vec4 height{ simulation.heightArray.read<glm::vec4>(index) };
		const glm::vec4 flow{ simulation.flowArray.read<glm::vec4>(index) };
		const glm::vec4 outFlux{ simulation.fluxArray.read<glm::vec4>(index) };
		glm::vec4 inFlux{ 0.0f };
		glm::vec4 inSedimentFlux{ 0.0f };

		struct Neighbor
		{
			glm::ivec3 index;
			glm::i8vec4 pipe;
			glm::vec4 flux;
			glm::vec4 sedimentFlux;
		};

		const Neighborhood neighborhood;
		
		for (int i{ 0 }; i < neighborhood.count; ++i)
		{
			Neighbor neighbor;
			neighbor.index.x = index.x + neighborhood.offsets[i].x;
			neighbor.index.y = index.y + neighborhood.offsets[i].y;

			if (isOutside(glm::ivec2{ neighbor.index }, glm::ivec2{ simulation.gridSize }))
			{
				continue;
			}

			const int direction{ (i + 2) % 4 };
			neighbor.index.z = 0;

			do
			{
				neighbor.pipe = simulation.pipeArray.read<glm::i8vec4>(neighbor.index);

				if (neighbor.pipe[direction] == index.z)
				{
					neighbor.flux = simulation.fluxArray.read<glm::vec4>(neighbor.index);
					neighbor.sedimentFlux += simulation.sedimentFluxArray.read<glm::vec4>(neighbor.index);

					inFlux[i] += neighbor.flux[direction];
					inSedimentFlux[i] = neighbor.sedimentFlux[direction];
				}

				neighbor.index.z = simulation.infoArray.read<glm::i8vec4>(neighbor.index)[ABOVE];
			}
			while (neighbor.index.z != INVALID_INDEX);
		}
		
		const float totalFlux{ inFlux[RIGHT] + inFlux[UP] + inFlux[LEFT] + inFlux[DOWN] - (outFlux[RIGHT] + outFlux[UP] + outFlux[LEFT] + outFlux[DOWN]) };
		const float totalSediment{ glm::length(outFlux) < 1e-6f ? glm::length(inSedimentFlux) + flow[SEDIMENT] : glm::length(inSedimentFlux)};

		const float waterHeight{ glm::min(height[WATER] + totalFlux * simulation.deltaTime * simulation.rGridScale * simulation.rGridScale, height[MAX_HEIGHT] - height[BEDROCK] - height[SAND]) };
		const float avgWaterHeight{ 0.5f * (waterHeight + height[WATER]) };
		const glm::vec2 velocity{ avgWaterHeight > 1e-4f ? 0.5f * glm::vec2{ inFlux[LEFT] - outFlux[LEFT] + outFlux[RIGHT] - inFlux[RIGHT],
																			 inFlux[DOWN] - outFlux[DOWN] + outFlux[UP] - inFlux[UP] } * simulation.deltaTime * simulation.gridScale * simulation.gridScale / avgWaterHeight : glm::vec2{ 0.0f } };
		height[WATER] = waterHeight;
		height[WATER] = glm::max((1.0f - simulation.evaporation * simulation.deltaTime) * height[WATER], 0.0f);

		simulation.heightArray.write(index, height);
		simulation.flowArray.write(index, glm::vec4{ velocity, totalSediment, 0.0f });

		index.z = simulation.infoArray.read<glm::i8vec4>(index)[ABOVE];
	}
	while (index.z != INVALID_INDEX);
}

void water(const Launch& launch, const Simulation& simulation)
{
	rainKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
	fluxKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
	waterKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
}

}
}
