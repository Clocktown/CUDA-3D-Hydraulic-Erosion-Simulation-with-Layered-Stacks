#pragma once

#include "../graphics/terrain.hpp"
#include "../device/simulation.hpp"
#include "../device/launch.hpp"
#include <onec/onec.hpp>

namespace geo
{

struct Simulation
{
	device::Simulation deviceParameters;
	device::Launch launchParameters;
	std::shared_ptr<Terrain> terrain{ nullptr };
	onec::cu::GraphicsResource infoResource;
	onec::cu::GraphicsResource heightResource;
	onec::cu::GraphicsResource waterVelocityResource;
	onec::cu::ArrayView infoArray;
	onec::cu::ArrayView heightArray;
	onec::cu::ArrayView waterVelocityArray;
	onec::cu::Surface infoSurface;
	onec::cu::Surface heightSurface;
	onec::cu::Surface waterVelocitySurface;
	bool isPaused{ false };
};

}
