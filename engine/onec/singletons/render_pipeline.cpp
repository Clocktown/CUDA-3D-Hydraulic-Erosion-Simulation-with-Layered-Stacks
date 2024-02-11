#include "render_pipeline.hpp"
#include "../core/world.hpp"

namespace onec
{

void RenderPipeline::render()
{
	World& world{ getWorld() };
	world.dispatch<OnRender>();
}

}

