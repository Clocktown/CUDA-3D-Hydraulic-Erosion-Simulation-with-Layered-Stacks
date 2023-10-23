#include "viewport_system.hpp"
#include "../config/config.hpp"
#include "../core/window.hpp"
#include "../core/scene.hpp"
#include "../singletons/viewport.hpp"
#include <glm/glm.hpp>

namespace onec
{

void ViewportSystem::update()
{
	const Window& window{ getWindow() };
	
	if (!window.isMinimized())
	{
		Scene& scene{ getScene() };

		ONEC_ASSERT(scene.hasSingleton<Viewport>(), "Scene must have a viewport singleton");

		Viewport& viewport{ *scene.getSingleton<Viewport>() };
		viewport.size = window.getFramebufferSize();
	}
}

}
