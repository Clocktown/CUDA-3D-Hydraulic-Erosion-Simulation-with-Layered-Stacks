#include "simulation_system.hpp"
#include "../components/simulation.hpp"
#include "../device/kernels.hpp"
#include <onec/onec.hpp>

namespace geo
{

template<typename ...Includes, typename ...Excludes>
void updateSimulation(const float deltaTime, const entt::exclude_t<Excludes...> excludes)
{
	onec::World& world{ onec::getWorld() };

	ONEC_ASSERT(world.hasSingleton<onec::Gravity>(), "World must have a gravity singleton");

	const auto view{ world.getView<Simulation, Includes...>(excludes) };
	const float gravity{ world.getSingleton<onec::Gravity>()->gravity.y };
	
	for (const entt::entity entity : view)
	{
		Simulation& simulation{ view.get<Simulation>(entity) };
		simulation.data.deltaTime = deltaTime;
		simulation.data.gravity = gravity;

		simulation.map();

		device::water(simulation.launch, simulation.data);

		simulation.unmap();
	}
}

}
