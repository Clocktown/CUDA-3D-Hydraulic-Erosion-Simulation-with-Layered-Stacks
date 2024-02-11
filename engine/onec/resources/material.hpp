#pragma once

#include "program.hpp"
#include "render_state.hpp"
#include "graphics_buffer.hpp"
#include <memory>

namespace onec
{

struct Material
{
	std::shared_ptr<Program> program{ nullptr };
	std::shared_ptr<RenderState> renderState{ nullptr };
	GraphicsBuffer uniformBuffer;
};

}
