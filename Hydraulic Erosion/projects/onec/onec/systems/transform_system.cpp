#include "transform_system.hpp"
#include <entt/entt.hpp>

namespace onec
{

void TransformSystem::start()
{
	updateLocalToParent();
	updateLocalToWorld();
}

void TransformSystem::update()
{
	updateLocalToParent(entt::exclude<Static>);
	updateLocalToWorld(entt::exclude<Static>);
}

}
