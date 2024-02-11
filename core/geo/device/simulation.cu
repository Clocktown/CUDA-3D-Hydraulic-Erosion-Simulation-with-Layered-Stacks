#include "simulation.hpp"

namespace geo
{
namespace device
{

__constant__ Simulation simulation{};
__constant__ int2 offsets[4]{ { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };

void setSimulation(const Simulation& data)
{
	CU_CHECK_ERROR(cudaMemcpyToSymbol(simulation, &data, sizeof(Simulation)));
}

}
}
