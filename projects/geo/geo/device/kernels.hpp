#pragma once

#include "simulation.hpp"
#include "launch.hpp"

namespace geo
{
namespace device
{

void initialization(const Launch& launch, const Simulation& simulation);
void water(const Launch& launch, const Simulation& simulation);

}
}
