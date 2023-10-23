#include "camera_system.hpp"
#include <entt/entt.hpp>

namespace onec
{

void CameraSystem::start()
{
	updateParentToView();
	updateWorldToView();
	updateViewToClip();
}

void CameraSystem::update()
{
	updateParentToView(entt::exclude<Static>);
	updateWorldToView(entt::exclude<Static>);
	updateViewToClip(entt::exclude<Static>);
}

}
