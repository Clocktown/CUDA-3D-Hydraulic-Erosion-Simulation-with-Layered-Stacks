#include "world.hpp"
#include "../events/world.hpp"
#include "../config/config.hpp"
#include <entt/entt.hpp>
#include <type_traits>
#include <memory>
#include <vector>

namespace onec
{

template<typename Event>
inline void World::dispatch(Event&& args)
{
	const auto& delegates{ getDelegates<std::decay_t<Event>>() };
	std::decay_t<Event> event{ args };

	for (const auto& system : delegates)
	{
		system(event);
	}
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
	entt::delegate<void(std::decay_t<Event>&)> delegate;
	delegate.connect<System>();

	auto& delegates{ getDelegates<std::decay_t<Event>>() };
	delegates.erase(std::find(delegates.begin(), delegates.end(), delegate), delegates.end());
	
	delegates.emplace_back(std::move(delegate));

	if constexpr (internal::OnCreateTrait<Event>::value)
	{
		using Type = typename internal::OnCreateTrait<Event>::type;
		m_registry.on_construct<Type>().connect<&World::onCreate<Type>>(*this);
	}

	if constexpr (internal::OnModifyTrait<Event>::value)
	{
		using Type = typename internal::OnModifyTrait<Event>::type;
		m_registry.on_update<Type>().connect<&World::onModify<Type>>(*this);
	}

	if constexpr (internal::OnDestroyTrait<Event>::value)
	{
		using Type = typename internal::OnDestroyTrait<Event>::type;
		m_registry.on_destroy<Type>().connect<&World::onDestroy<Type>>(*this);
	}
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
	entt::delegate<void(std::decay_t<Event>&)> delegate;
	delegate.connect<system>();

	auto& delegates{ getDelegates<std::decay_t<Event>>() };
	delegates.erase(std::find(delegates.begin(), delegates.end(), delegate));
}

template<typename Event>
inline void World::removeSystems()
{
	m_dispatcher.erase(entt::type_hash<std::decay_t<Event>>::value());
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

template<typename Type>
inline void World::onCreate([[maybe_unused]] const entt::registry& registry, const entt::entity entity)
{
	dispatch(OnCreate<Type>{ entity });
}

template<typename Type>
inline void World::onModify([[maybe_unused]] const entt::registry& registry, const entt::entity entity)
{
	dispatch(OnModify<Type>{ entity });
}

template<typename Type>
inline void World::onDestroy([[maybe_unused]] const entt::registry& registry, const entt::entity entity)
{
	dispatch(OnDestroy<Type>{ entity });
}

template<typename Event>
inline const std::vector<entt::delegate<void(Event&)>>* World::getDelegates() const
{
	static_assert(std::is_same_v<Event, std::decay_t<Event>>, "Event must be decayed");

	const auto iterator{ m_dispatcher.find(entt::type_hash<Event>::value()) };

	if (iterator != m_dispatcher.end())
	{
		return static_cast<std::vector<entt::delegate<void(Event&)>>*>(iterator->second.get());
	}

	return nullptr;
}

template<typename Event>
inline std::vector<entt::delegate<void(Event&)>>& World::getDelegates()
{
	static_assert(std::is_same_v<Event, std::decay_t<Event>>, "Event must be decayed");

	std::shared_ptr<void>& pointer{ m_dispatcher[entt::type_hash<Event>::value()] };

	if (pointer == nullptr)
	{
		pointer = std::make_shared<std::vector<entt::delegate<void(Event&)>>>();
	}

	return *static_cast<std::vector<entt::delegate<void(Event&)>>*>(pointer.get());
}

}
