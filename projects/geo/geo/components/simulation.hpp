#pragma once

#include "../graphics/terrain.hpp"
#include "../device/simulation.hpp"
#include "../device/launch.hpp"

namespace geo
{

struct Simulation
{
	explicit Simulation() = default;
	explicit Simulation(const std::shared_ptr<Terrain>& terrain);

	void map();
	void unmap();

	std::shared_ptr<Terrain> terrain;
	device::Simulation data;
	device::Launch launch;
	onec::cu::Array infoArray;
	onec::cu::Array heightArray;
	onec::cu::Array sedimentArray;
	onec::cu::Array pipeArray;
	onec::cu::Array fluxArray;
	onec::cu::Array sedimentFluxArray;
	onec::cu::Array velocityArray;
};

}
