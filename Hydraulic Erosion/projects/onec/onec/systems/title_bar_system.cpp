#include "title_bar_system.hpp"
#include "../core/application.hpp"
#include "../core/window.hpp"
#include <string>

namespace onec
{

void TitleBarSystem::update()
{
	const Application& application{ getApplication() };
	Window& window{ getWindow() };
	
	const std::string fps{ std::to_string(application.getFrameRate()) + "fps" };
	const std::string ms{ std::to_string(1000.0 * application.getUnscaledDeltaTime()) + "ms" };
	const std::string title{ application.getName() + " @ " + fps + " / " + ms};

	window.setTitle(title);;
}

}
