#include "world.hpp"
#include "../config/config.hpp"
#include <entt/entt.hpp>

namespace onec
{

World::World(const std::string_view& name) :
	m_name{ name },
	m_singleton{ m_registry.create() }
{
	
}

entt::entity World::addEntity()
{
	return m_registry.create();
}

void World::removeEntity(const entt::entity entity)
{
	ONEC_ASSERT(entity != m_singleton, "Singleton cannot be destroyed");

	m_registry.destroy(entity);
}

void World::removeEntities()
{
	m_registry.clear();
	m_singleton = m_registry.create();
}

void World::removeSystems()
{
	m_systems.clear();
}

void World::setName(const std::string_view& name)
{
	m_name = name;
}

const std::string& World::getName() const
{
	return m_name;
}

entt::entity World::getSingleton() const
{
	return m_singleton;
}

int World::getEntityCount() const
{
	return static_cast<int>(m_registry.storage<entt::entity>()->in_use());
}

bool World::hasEntities() const
{
	return m_registry.storage<entt::entity>()->empty();
}

bool World::hasEntity(const entt::entity entity) const
{
	auto* entities{ m_registry.storage<entt::entity>() };
	return entities->contains(entity) && (entities->index(entity) < entities->in_use());
}

World& getWorld()
{
	static World world;
	return world;
}

}
