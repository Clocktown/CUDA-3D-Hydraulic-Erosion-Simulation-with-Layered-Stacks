#include "screenshot.hpp"
#include "../config/config.hpp"
#include "../config/gl.hpp"
#include "../core/world.hpp"
#include "../singletons/viewport.hpp"
#include "../utility/io.hpp"
#include "../utility/time.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <filesystem>
#include <string>
#include <vector>

namespace onec
{

void takeScreenshot(const std::string_view name, const std::filesystem::path& folder)
{
	World& world{ getWorld() };
	
	ONEC_ASSERT(world.hasSingleton<Viewport>(), "World must have a viewport singleton");

	const Viewport viewport{ *world.getSingleton<Viewport>() };
	std::vector<unsigned char> screenshot(4 * static_cast<std::size_t>(viewport.size.x) * static_cast<std::size_t>(viewport.size.y));

	GL_CHECK_ERROR(glReadPixels(viewport.offset.x, viewport.offset.y, viewport.size.x, viewport.size.y, GL_RGBA, GL_UNSIGNED_BYTE, screenshot.data()));

	writeImage(folder / (std::string{ name } + " " + formatDateTime(getDateTime()) + ".png"), asBytes(screenshot), viewport.size, 4);
}

}
