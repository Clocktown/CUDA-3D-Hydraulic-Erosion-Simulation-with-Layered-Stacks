#include "scene.hpp"
#include "../config/config.hpp"
#include <entt/entt.hpp>
#include <cassert>
#include <memory>
#include <type_traits>

namespace onec
{

template<typename Event>
inline void Scene::dispatch(Event&& event)
{
	const auto& systems{ getSystems<std::decay_t<Event>>() };

	for (const auto& system : systems)
	{
		system(event);
	}
}

template<typename Type, typename... Args>
inline decltype(auto) Scene::addComponent(const entt::entity entity, Args&&... args)
{
	return m_registry.get_or_emplace<Type>(entity, std::forward<Args>(args)...);
}

template<typename Type, typename... Args>
inline decltype(auto) Scene::addSingleton(Args&&... args)
{
	return m_registry.get_or_emplace<Type>(m_singleton, std::forward<Args>(args)...);
}

template<typename Event, auto system, typename... Instance>
inline void Scene::addSystem(Instance&&... instance)
{
	auto& systems{ getSystems<std::decay_t<Event>>() };
	systems.emplace_back().connect<system>(std::forward<Instance>(instance)...);

	if constexpr (internal::OnCreateTrait<Event>::value)
	{
		using Type = typename internal::OnCreateTrait<Event>::type;
		m_registry.on_construct<Type>().connect<&Scene::onCreate<Type>>(*this);
	}

	if constexpr (internal::OnModifyTrait<Event>::value)
	{
		using Type = typename internal::OnModifyTrait<Event>::type;
		m_registry.on_update<Type>().connect<&Scene::onModify<Type>>(*this);
	}

	if constexpr (internal::OnDestroyTrait<Event>::value)
	{
		using Type = typename internal::OnDestroyTrait<Event>::type;
		m_registry.on_destroy<Type>().connect<&Scene::onDestroy<Type>>(*this);
	}
}

template<typename Type>
inline void Scene::removeComponent(const entt::entity entity)
{
	auto& components{ m_registry.storage<Type>() };

	if (components.contains(entity))
	{
		components.erase(entity);
	}
}

template<typename Type>
inline void Scene::removeComponents()
{
	m_registry.clear<Type>();
}

template<typename Type>
inline void Scene::removeSingleton()
{
	auto& components{ m_registry.storage<Type>() };

	if (components.contains(m_singleton))
	{
		components.erase(m_singleton);
	}
}

template<typename Event, auto system, typename... Instance>
inline void Scene::removeSystem(Instance&&... instance)
{
	const entt::delegate<void(std::decay_t<Event>&)> delegate;
	delegate.connect<system>(std::forward<Instance>(instance)...);

	auto& systems{ getSystems<std::decay_t<Event>>() };
	std::remove(systems.begin(), systems.end(), delegate);
}

template<typename Event>
inline void Scene::removeSystems()
{
	m_systems.erase(entt::type_hash<std::decay_t<Event>>::value());
}

template<typename Type>
inline void Scene::onCreate([[maybe_unused]] const entt::registry& registry, const entt::entity entity)
{
	dispatch(OnCreate<Type>{ entity });
}

template<typename Type>
inline void Scene::onModify([[maybe_unused]] const entt::registry& registry, const entt::entity entity)
{
	dispatch(OnModify<Type>{ entity });
}

template<typename Type>
inline void Scene::onDestroy([[maybe_unused]] const entt::registry& registry, const entt::entity entity)
{
	dispatch(OnDestroy<Type>{ entity });
}

template<typename Type, typename... Args>
inline decltype(auto) Scene::setComponent(const entt::entity entity, Args&&... args)
{
	return m_registry.emplace_or_replace<Type>(entity, std::forward<Args>(args)...);
}

template<typename Type, typename... Args>
inline decltype(auto) Scene::setSingleton(Args&&... args)
{
	return m_registry.emplace_or_replace<Type>(m_singleton, std::forward<Args>(args)...);
}

template<typename Type>
inline entt::entity Scene::getEntity(Type& component)
{
	return entt::to_entity(m_registry, component);
}

template<typename Type, typename... Others, typename... Excludes>
inline auto Scene::getView(const entt::exclude_t<Excludes...> excludes) const
{
	return m_registry.view<Type, Others...>(excludes);
}

template<typename Type, typename... Others, typename... Excludes>
inline auto Scene::getView(const entt::exclude_t<Excludes...> excludes)
{
	return m_registry.view<Type, Others...>(excludes);
}

template<typename Type>
inline const Type* Scene::getComponent(const entt::entity entity) const
{
	return m_registry.try_get<Type>(entity);
}

template<typename Type>
inline Type* Scene::getComponent(const entt::entity entity)
{
	return m_registry.try_get<Type>(entity);
}

template<typename Type>
inline const Type* Scene::getSingleton() const
{
	return m_registry.try_get<Type>(m_singleton);
}

template<typename Type>
inline Type* Scene::getSingleton()
{
	return m_registry.try_get<Type>(m_singleton);
}

template<typename Event>
inline const auto* Scene::getSystems() const
{
	static_assert(std::is_same_v<Event, std::decay_t<Event>>, "Event must be decayed");

	const auto it{ m_systems.find(entt::type_hash<Event>::value()) };

	if (it != m_systems.end())
	{
		return static_cast<std::vector<entt::delegate<void(Event&)>>*>(it->second.get());
	}

	return nullptr;
}

template<typename Event>
inline auto& Scene::getSystems()
{
	static_assert(std::is_same_v<Event, std::decay_t<Event>>, "Event must be decayed");

	auto& pointer = m_systems[entt::type_hash<Event>::value()];

	if (pointer == nullptr)
	{
		pointer = std::make_shared<std::vector<entt::delegate<void(Event&)>>>();
	}

	return *static_cast<std::vector<entt::delegate<void(Event&)>>*>(pointer.get());
}

template<typename Type>
inline bool Scene::hasComponent(const entt::entity entity) const
{
	return m_registry.any_of<Type>(entity);
}

template<typename Type>
inline bool Scene::hasSingleton() const
{
	return m_registry.any_of<Type>(m_singleton);
}

}
