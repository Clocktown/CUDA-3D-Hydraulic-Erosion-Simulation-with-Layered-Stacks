#include "kernels.hpp"
#include "common.hpp"
#include <onec/config/cu.hpp>
#include <onec/cu/launch.hpp>
#include <onec/utility/grid.hpp>
#include <float.h>

namespace geo
{
namespace device
{

__global__ void initializationKernel(Simulation simulation)
{
	glm::ivec3 index{ glm::ivec2{ onec::cu::getGlobalIndex() }, 0 };

	if (onec::isOutside(glm::ivec2{ index }, glm::ivec2{ simulation.gridSize }))
	{
		return;
	}

	simulation.heightArray.write(index, glm::vec4{ (simulation.gridSize.x - index.x) / 16.0f, 0.0f, 0.0f, index.x > 64 ? 30.0f : FLT_MAX });
	simulation.infoArray.write(index, glm::i8vec4{ INVALID_INDEX, index.x > 64 ? 1 : INVALID_INDEX, 0, 0 });
	simulation.fluxArray.write(index, glm::vec4{ 0.0f });

	index.z = 1;
	simulation.heightArray.write(index, glm::vec4{ (index.x > 64) * (30.0f + index.x / 16.0f), 0.0f, (index.x > 64) * 10.0f, FLT_MAX });
	simulation.infoArray.write(index, glm::i8vec4{ index.x > 64 ? 0 : INVALID_INDEX, INVALID_INDEX, 0, 0 });
	simulation.fluxArray.write(index, glm::vec4{ 0.0f });

	for (index.z = 2; index.z < simulation.gridSize.z; ++index.z)
	{
		simulation.heightArray.write(index, glm::vec4{ 0.0f, 0.0f, 0.0f, FLT_MAX });
		simulation.infoArray.write(index, glm::i8vec4{ INVALID_INDEX, INVALID_INDEX, 0, 0 });
		simulation.fluxArray.write(index, glm::vec4{ 0.0f });
	}
}

void initialization(const Launch& launch, const Simulation& simulation) 
{
	initializationKernel<<<launch.gridSize, launch.blockSize>>>(simulation);
}

}
}
