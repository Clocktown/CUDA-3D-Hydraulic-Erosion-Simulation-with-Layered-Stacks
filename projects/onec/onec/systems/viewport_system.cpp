#include "viewport_system.hpp"
#include "../config/config.hpp"
#include "../core/window.hpp"
#include "../core/world.hpp"
#include "../singletons/viewport.hpp"
#include <glm/glm.hpp>

namespace onec
{

void updateViewport()
{
	const Window& window{ getWindow() };
	
	if (!window.isMinimized())
	{
		World& world{ getWorld() };

		ONEC_ASSERT(world.hasSingleton<Viewport>(), "World must have a viewport singleton");

		Viewport& viewport{ *world.getSingleton<Viewport>() };
		viewport.size = window.getFramebufferSize();
	}
}

}
