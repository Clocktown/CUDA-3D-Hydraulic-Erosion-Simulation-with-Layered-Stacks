#include "gui_system.hpp"
#include "../singletons/gui.hpp"
#include <onec/onec.hpp>

using namespace onec;

namespace geo
{

void GUISystem::start()
{
	Scene& scene{ getScene() };

	ONEC_ASSERT(scene.hasSingleton<GUI>(), "Scene must have a gui singleton");
	ONEC_ASSERT(scene.hasSingleton<Renderer>(), "Scene must have a renderer singleton");

	GUI& gui{ *scene.getSingleton<GUI>() };
	Renderer& renderer{ *scene.getSingleton<Renderer>() };

	renderer.clearColor = glm::vec4{ gui.rendering.backgroundColor, 1.0f };
}

void GUISystem::update()
{

}

}
