#include "world.hpp"
#include "../config/config.hpp"
#include <entt/entt.hpp>
#include <type_traits>

namespace onec
{

template<typename Event, typename... Args>
inline void World::dispatch(Args&&... args)
{
	m_dispatcher.trigger(Event{ args... });
}

template<typename Type, typename... Args>
inline decltype(auto) World::addComponent(const entt::entity entity, Args&&... args)
{
	return m_registry.get_or_emplace<Type>(entity, std::forward<Args>(args)...);
}

template<typename Type, typename... Args>
inline decltype(auto) World::addSingleton(Args&&... args)
{
	if constexpr (std::is_empty_v<Type>)
	{
		m_registry.ctx().emplace<Type>(std::forward<Args>(args)...);
	}
	else
	{
		return m_registry.ctx().emplace<Type>(std::forward<Args>(args)...);
	}
}

template<typename Event, auto System, typename... Instance>
inline void World::addSystem(Instance&&... instance)
{
	m_dispatcher.sink<Event>().connect<System>(std::forward<Instance>(instance)...);
}

template<typename Type>
inline void World::removeComponent(const entt::entity entity)
{
	auto& storage{ m_registry.storage<Type>() };

	if (storage.contains(entity))
	{
		storage.erase(entity);
	}
}

template<typename Type>
inline void World::removeComponents()
{
	m_registry.clear<Type>();
}

template<typename Type>
inline void World::removeSingleton()
{
	m_registry.ctx().erase<Type>();
}

template<typename Event, auto System, typename... Instance>
inline void World::removeSystem(Instance&&... instance)
{
	m_dispatcher.sink<Event>().disconnect<System>(std::forward<Instance>(instance)...);
}

template<typename Event>
inline void World::removeSystems()
{
	m_dispatcher.sink<Event>().clear();
}

template<typename Type, typename... Args>
inline decltype(auto) World::setComponent(const entt::entity entity, Args&&... args)
{
	return m_registry.emplace_or_replace<Type>(entity, std::forward<Args>(args)...);
}

template<typename Type, typename... Args>
inline decltype(auto) World::setSingleton(Args&&... args)
{
	if constexpr (std::is_empty_v<Type>)
	{
		m_registry.ctx().emplace<Type>(std::forward<Args>(args)...);
	}
	else
	{
		return m_registry.ctx().insert_or_assign(Type{ args... });
	}
	
}

template<typename Type>
inline const entt::registry::storage_for_type<Type>* World::getStorage() const
{
	return m_registry.storage<Type>();
}

template<typename Type>
inline entt::registry::storage_for_type<Type>& World::getStorage()
{
	return m_registry.storage<Type>();
}

template<typename Type, typename... Others, typename... Excludes>
inline entt::view<entt::get_t<const Type, const Others...>, entt::exclude_t<const Excludes...>> World::getView(const entt::exclude_t<Excludes...> excludes) const
{
	return m_registry.view<Type, Others...>(excludes);
}

template<typename Type, typename... Others, typename... Excludes>
inline entt::view<entt::get_t<Type, Others...>, entt::exclude_t<Excludes...>> World::getView(const entt::exclude_t<Excludes...> excludes)
{
	return m_registry.view<Type, Others...>(excludes);
}

template<typename Type>
inline entt::entity World::getEntity(const Type& component) const
{
	return entt::to_entity(m_registry, component);
}

template<typename Type>
inline const Type* World::getComponent(const entt::entity entity) const
{
	return m_registry.try_get<Type>(entity);
}

template<typename Type>
inline Type* World::getComponent(const entt::entity entity)
{
	return m_registry.try_get<Type>(entity);
}

template<typename Type>
inline const Type* World::getSingleton() const
{
	return m_registry.ctx().find<Type>();
}

template<typename Type>
inline Type* World::getSingleton()
{
	return m_registry.ctx().find<Type>();
}

template<typename Type>
inline bool World::hasComponent(const entt::entity entity) const
{
	return m_registry.all_of<Type>(entity);
}

template<typename Type>
inline bool World::hasSingleton() const
{
	return m_registry.ctx().contains<Type>();
}

}
