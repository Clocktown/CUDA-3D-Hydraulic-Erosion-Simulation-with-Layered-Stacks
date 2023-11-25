#include "simulation_system.hpp"
#include "../components/simulation.hpp"
#include "../device/kernels.hpp"
#include <onec/onec.hpp>

namespace geo
{

void SimulationSystem::initialize(Simulation& simulation)
{
	const int device{ 0 };
	int smCount;
	int smThreadCount;
	CU_CHECK_ERROR(cudaSetDevice(device));
	CU_CHECK_ERROR(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
	CU_CHECK_ERROR(cudaDeviceGetAttribute(&smThreadCount, cudaDevAttrMaxThreadsPerMultiProcessor, device));
	const float threadCount{ static_cast<float>(smCount * smThreadCount) };

	simulation.launchParameters.standard1D.gridSize = static_cast<unsigned int>(glm::ceil(static_cast<float>(simulation.deviceParameters.horizontalCellCount) / static_cast<float>(simulation.launchParameters.standard1D.blockSize)));
	simulation.launchParameters.standard2D.gridSize.x = static_cast<unsigned int>(glm::ceil(static_cast<float>(simulation.deviceParameters.gridSize.x) / static_cast<float>(simulation.launchParameters.standard2D.blockSize.x)));
	simulation.launchParameters.standard2D.gridSize.y = static_cast<unsigned int>(glm::ceil(static_cast<float>(simulation.deviceParameters.gridSize.y) / static_cast<float>(simulation.launchParameters.standard2D.blockSize.y)));
	simulation.launchParameters.standard2D.gridSize.z = 1;
	simulation.launchParameters.standard2D.blockSize.z = 1;
	simulation.launchParameters.standard3D.gridSize.x = static_cast<unsigned int>(glm::ceil(static_cast<float>(simulation.deviceParameters.gridSize.x) / static_cast<float>(simulation.launchParameters.standard2D.blockSize.x)));
	simulation.launchParameters.standard3D.gridSize.y = static_cast<unsigned int>(glm::ceil(static_cast<float>(simulation.deviceParameters.gridSize.y) / static_cast<float>(simulation.launchParameters.standard2D.blockSize.y)));
	simulation.launchParameters.standard3D.gridSize.z = static_cast<unsigned int>(glm::ceil(static_cast<float>(simulation.deviceParameters.gridSize.z) / static_cast<float>(simulation.launchParameters.standard2D.blockSize.z)));

	simulation.launchParameters.gridStride1D.gridSize = static_cast<unsigned int>(threadCount / static_cast<float>(simulation.launchParameters.gridStride1D.blockSize));
	simulation.launchParameters.gridStride2D.gridSize.x = onec::nextPowerOfTwo(static_cast<unsigned int>(glm::ceil(glm::sqrt(threadCount / static_cast<float>(simulation.launchParameters.gridStride2D.blockSize.x * simulation.launchParameters.gridStride2D.blockSize.y)))));
	simulation.launchParameters.gridStride2D.gridSize.y = simulation.launchParameters.gridStride2D.gridSize.x;
	simulation.launchParameters.gridStride2D.gridSize.z = 1;
	simulation.launchParameters.gridStride2D.blockSize.z = 1;
	simulation.launchParameters.gridStride3D.gridSize.x = onec::nextPowerOfTwo(static_cast<unsigned int>(glm::ceil(glm::pow(threadCount / static_cast<float>(simulation.launchParameters.gridStride2D.blockSize.x * simulation.launchParameters.gridStride2D.blockSize.y * simulation.launchParameters.gridStride2D.blockSize.z), 1.0f / 3.0f))));
	simulation.launchParameters.gridStride3D.gridSize.y = simulation.launchParameters.gridStride3D.gridSize.x;
	simulation.launchParameters.gridStride3D.gridSize.z = simulation.launchParameters.gridStride3D.gridSize.x;

	ONEC_ASSERT(simulation.terrain != nullptr, "Terrain cannot be nullptr");

	Terrain& terrain{ *simulation.terrain };
	simulation.deviceParameters.gridSize = glm::ivec3{ terrain.gridSize, terrain.maxLayerCount };
	simulation.deviceParameters.gridScale = terrain.gridScale;
	simulation.deviceParameters.rGridScale = 1.0f / terrain.gridScale;
	simulation.deviceParameters.cellCount = terrain.gridSize.x * terrain.gridSize.y * terrain.maxLayerCount;
	simulation.deviceParameters.horizontalCellCount = terrain.gridSize.x * terrain.gridSize.y;

	simulation.infoResource.initialize(terrain.infoMap, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	simulation.heightResource.initialize(terrain.heightMap, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	simulation.waterVelocityResource.initialize(terrain.waterVelocityMap, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	map(simulation);

	device::initialize(simulation.launchParameters, simulation.deviceParameters);

	unmap(simulation);
}

void SimulationSystem::update()
{
	const onec::Application& application{ onec::getApplication() };
	onec::World& world{ onec::getWorld() };

	const auto view{ world.getView<Simulation>() };

	for (const entt::entity entity : view)
	{
		simulate(view.get<Simulation>(entity), static_cast<float>(application.getDeltaTime()));
	}
}

void SimulationSystem::fixedUpdate()
{
	const onec::Application& application{ onec::getApplication() };
	onec::World& world{ onec::getWorld() };

	const auto view{ world.getView<Simulation>() };

	for (const entt::entity entity : view)
	{
		simulate(view.get<Simulation>(entity), static_cast<float>(application.getFixedDeltaTime()));
	}
}

void SimulationSystem::simulate(Simulation& simulation, const float deltaTime)
{
	if (simulation.isPaused)
	{
		return;
	}

	map(simulation);

	{
		onec::World& world{ onec::getWorld() };

		ONEC_ASSERT(world.hasSingleton<onec::Gravity>(), "World must have a gravity singleton");

		simulation.deviceParameters.deltaTime = deltaTime;
		simulation.deviceParameters.gravity = world.getSingleton<onec::Gravity>()->gravity;
	}

	unmap(simulation);
}

void SimulationSystem::map(Simulation& simulation)
{
	simulation.infoResource.map();
	simulation.heightResource.map();
	simulation.waterVelocityResource.map();

	simulation.infoArray = simulation.infoResource.getArrayView();
	simulation.heightArray = simulation.heightResource.getArrayView();
	simulation.waterVelocityArray = simulation.waterVelocityResource.getArrayView();

	simulation.infoSurface.initialize(simulation.infoArray);
	simulation.heightSurface.initialize(simulation.heightArray);
	simulation.waterVelocitySurface.initialize(simulation.waterVelocityArray);

	simulation.deviceParameters.infoSurface = simulation.infoSurface.getHandle();
	simulation.deviceParameters.heightSurface = simulation.heightSurface.getHandle();
	simulation.deviceParameters.waterVelocitySurface = simulation.waterVelocitySurface.getHandle();
}

void SimulationSystem::unmap(Simulation& simulation)
{
	simulation.infoResource.unmap();
	simulation.heightResource.unmap();
	simulation.waterVelocityResource.unmap();
}

}
