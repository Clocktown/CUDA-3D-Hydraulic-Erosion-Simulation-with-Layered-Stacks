#pragma once

#include "program.hpp"
#include "render_state.hpp"
#include "buffer.hpp"
#include <memory>

namespace onec
{

struct Material
{
	std::shared_ptr<Program> program{ nullptr };
	std::shared_ptr<RenderState> renderState{ nullptr };
	Buffer uniformBuffer;
};

}
