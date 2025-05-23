#include "simulation.hpp"

namespace geo
{
namespace device
{

__constant__ Simulation simulation;
__constant__ float2 sceneBounds;
__constant__ int2 offsets[4]{ { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };

void setSimulation(const Simulation& simulation)
{
	CU_CHECK_ERROR(cudaMemcpyToSymbol(device::simulation, &simulation, sizeof(Simulation)));
}

void setSceneBounds(const float2& bounds)
{
	CU_CHECK_ERROR(cudaMemcpyToSymbol(device::sceneBounds, &bounds, sizeof(float2)));
}

}
}
