#include "screenshot_system.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../core/input.hpp"
#include "../core/scene.hpp"
#include "../singletons/screenshot.hpp"
#include "../singletons/viewport.hpp"
#include "../utility/io.hpp"
#include <glad/glad.h>
#include <filesystem>
#include <vector>

namespace onec
{

void ScreenshotSystem::update()
{
	Input& input{ getInput() };
	
	if (input.isKeyReleased(GLFW_KEY_PRINT_SCREEN)) // GLFW reports GLFW_RELEASE falsely directly after GLFW_PRESS (only for GLFW_KEY_PRINT_SCREEN)
	{
		Scene& scene{ getScene() };

		ONEC_ASSERT(scene.hasSingleton<Screenshot>(), "Scene must have a screenshot singleton");
		ONEC_ASSERT(scene.hasSingleton<Viewport>(), "Scene must have a viewport singleton");

		const Screenshot& screenshot{ *scene.getSingleton<Screenshot>() };
		const Viewport& viewport{ *scene.getSingleton<Viewport>() };

		const glm::ivec2 size{ viewport.size };
		std::vector<unsigned char> data(4 * static_cast<size_t>(size.x) * static_cast<size_t>(size.x));

		GL_CHECK_ERROR(glReadPixels(viewport.offset.x, viewport.offset.y, size.x, size.y, GL_RGBA, GL_UNSIGNED_BYTE, data.data()));
		
		writeImage(screenshot.folder / (screenshot.name + " " + getDateTime() + ".png"), asBytes(data), size, 4);
	}
}

}
