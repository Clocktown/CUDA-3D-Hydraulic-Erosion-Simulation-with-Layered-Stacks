#include "scene.hpp"
#include "../config/config.hpp"
#include <entt/entt.hpp>

namespace onec
{

Scene::Scene(const std::string_view& name) :
	m_name{ name },
	m_singleton{ m_registry.create() }
{
	
}

entt::entity Scene::addEntity()
{
	return m_registry.create();
}

void Scene::removeEntity(const entt::entity entity)
{
	ONEC_ASSERT(entity != m_singleton, "Singleton cannot be destroyed");

	m_registry.destroy(entity);
}

void Scene::removeEntities()
{
	m_registry.clear();
	m_singleton = m_registry.create();
}

void Scene::removeSystems()
{
	m_systems.clear();
}

void Scene::setName(const std::string_view& name)
{
	m_name = name;
}

const std::string& Scene::getName() const
{
	return m_name;
}

entt::entity Scene::getSingleton() const
{
	return m_singleton;
}

int Scene::getEntityCount() const
{
	return static_cast<int>(m_registry.storage<entt::entity>()->in_use());
}

bool Scene::hasEntities() const
{
	return m_registry.storage<entt::entity>()->empty();
}

bool Scene::hasEntity(const entt::entity entity) const
{
	auto* entities{ m_registry.storage<entt::entity>() };
	return entities->contains(entity) && (entities->index(entity) < entities->in_use());
}

Scene& getScene()
{
	static Scene scene;
	return scene;
}

}
