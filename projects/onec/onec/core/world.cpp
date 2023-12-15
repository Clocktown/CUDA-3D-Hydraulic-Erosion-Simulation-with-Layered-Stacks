#include "world.hpp"
#include "../config/config.hpp"
#include <entt/entt.hpp>
#include <xmemory>
#include <string>


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
	m_dispatcher.clear();
}

int World::getEntityCount() const
{
	return static_cast<int>(m_registry.storage<entt::entity>()->in_use());
}

bool World::hasEntities() const
{
	return m_registry.storage<entt::entity>()->in_use() > 0;
}

bool World::hasEntity(const entt::entity entity) const
{
	auto* storage{ m_registry.storage<entt::entity>() };
	return storage->contains(entity) && storage->index(entity) < storage->in_use();
}

World& getWorld()
{
	static World world;
	return world;
}

}
