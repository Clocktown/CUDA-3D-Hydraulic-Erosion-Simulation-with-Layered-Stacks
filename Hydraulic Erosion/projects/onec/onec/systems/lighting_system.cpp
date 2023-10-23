#include "lighting_system.hpp"
#include "../config/config.hpp"
#include "../core/scene.hpp"
#include "../singletons/lighting.hpp"
#include "../device/lighting.hpp"
#include "../graphics/buffer.hpp"

namespace onec
{

void LightingSystem::start()
{
	Scene& scene{ getScene() };

	ONEC_ASSERT(scene.hasSingleton<Lighting>(), "Scene must have a lighting singleton");

	Lighting& lighting{ *scene.getSingleton<Lighting>() };
	lighting.uniformBuffer.initialize(sizeof(device::Lighting));
}

void LightingSystem::update()
{
	updateUniformBuffer();
}

}
