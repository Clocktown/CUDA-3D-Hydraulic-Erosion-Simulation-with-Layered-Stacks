#include "simulation_system.hpp"
#include "../components/simulation.hpp"
#include "../singletons/rain.hpp"
#include "../device/kernels.hpp"
#include <onec/onec.hpp>

namespace geo
{

template<typename ...Includes, typename ...Excludes>
void simulate(const float deltaTime, const entt::exclude_t<Excludes...> excludes)
{
	onec::World& world{ onec::getWorld() };

	ONEC_ASSERT(world.hasSingleton<onec::Gravity>(), "World must have a gravity singleton");
	ONEC_ASSERT(world.hasSingleton<geo::Rain>(), "World must have a rain singleton");

	const float gravity{ world.getSingleton<onec::Gravity>()->gravity.y };
	const float rain{ world.getSingleton<geo::Rain>()->rain };

	const auto view{ world.getView<Simulation, Includes...>(excludes) };

	for (const entt::entity entity : view)
	{
		Simulation& simulation{ view.get<Simulation>(entity) };
		simulation.data.deltaTime = deltaTime;
		simulation.data.gravity = gravity;
		simulation.data.rain = rain;

		simulation.map();

		device::rain(simulation.launch, simulation.data);

		simulation.unmap();
	}
}

}
