#include "world.hpp"
#include "../config/config.hpp"
#include <entt/entt.hpp>

namespace onec
{

entt::entity World::addEntity()
{
	return m_registry.create();
}

void World::removeEntity(const entt::entity entity)
{
	m_registry.destroy(entity);
}

void World::removeEntities()
{
	m_registry.clear();
}

void World::removeSingletons()
{
	m_registry.ctx() = entt::internal::registry_context<std::allocator<entt::entity>>{ std::allocator<entt::entity>{} };
}

void World::removeSystems()
{
	m_dispatcher = entt::dispatcher{};
}

std::ptrdiff_t World::getEntityCount() const
{
	return static_cast<std::ptrdiff_t>(m_registry.storage<entt::entity>()->free_list());
}

bool World::hasEntities() const
{
	return m_registry.storage<entt::entity>()->free_list() > 0;
}

bool World::hasEntity(const entt::entity entity) const
{
	return m_registry.valid(entity);
}

World& getWorld()
{
	static World world;
	return world;
}

}
