#include "lighting_system.hpp"
#include "../config/config.hpp"
#include "../core/world.hpp"
#include "../singletons/lighting.hpp"
#include "../device/lighting.hpp"
#include "../graphics/buffer.hpp"

namespace onec
{

void LightingSystem::start()
{
	World& world{ getWorld() };

	ONEC_ASSERT(world.hasSingleton<Lighting>(), "World must have a lighting singleton");

	Lighting& lighting{ *world.getSingleton<Lighting>() };
	lighting.uniformBuffer.initialize(sizeof(device::Lighting));
}

}
