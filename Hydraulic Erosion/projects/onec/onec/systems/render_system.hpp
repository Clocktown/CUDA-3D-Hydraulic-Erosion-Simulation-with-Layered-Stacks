#pragma once

#include "../singletons/renderer.hpp"
#include "../singletons/viewport.hpp"
#include <entt/entt.hpp>

namespace onec
{

class RenderSystem
{
public:
	static void start();
	static void update();
private:
	static void updateUniformBuffer(Renderer& renderer, const Viewport& viewport, const entt::entity camera);
	static void updateRenderTarget(const Renderer& renderer, const Viewport viewport);
};

}
