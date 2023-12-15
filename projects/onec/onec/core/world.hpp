#pragma once

#include <entt/entt.hpp>
#include <memory>
#include <string>
#include <vector>

namespace onec
{

class World
{
public:
	World(const World& other) = delete;
	World(World&& other) = delete;

	~World() = default;

	World& operator=(const World& other) = delete;
	World& operator=(World&& other) = delete;

	entt::entity addEntity();
	void removeEntity(entt::entity entity);
	void removeEntities();
	void removeSingletons();
	void removeSystems();

	int getEntityCount() const;
	bool hasEntities() const;
	bool hasEntity(entt::entity entity) const;

	template<typename Event>
	void dispatch(Event&& event = Event{});

	template<typename Type, typename... Args>
	decltype(auto) addComponent(entt::entity entity, Args&&... args);

	template<typename Type, typename... Args>
	decltype(auto) addSingleton(Args&&... args);

	template<typename Event, auto System, typename... Instance>
	void addSystem(Instance&&... instance);

	template<typename Type>
	void removeComponent(entt::entity entity);

	template<typename Type>
	void removeComponents();

	template<typename Type>
	void removeSingleton();

	template<typename Event, auto System, typename... Instance>
	void removeSystem(Instance&&... instance);

	template<typename Event>
	void removeSystems();

	template<typename Type, typename... Args>
	decltype(auto) setComponent(entt::entity entity, Args&&... args);

	template<typename Type, typename... Args>
	decltype(auto) setSingleton(Args&&... args);

	template<typename Type>
	const entt::registry::storage_for_type<Type>* getStorage() const;

	template<typename Type>
	entt::registry::storage_for_type<Type>& getStorage();

	template<typename Type, typename... Others, typename... Excludes>
	entt::view<entt::get_t<const Type, const Others...>, entt::exclude_t<const Excludes...>> getView(entt::exclude_t<Excludes...> excludes = entt::exclude_t{}) const;

	template<typename Type, typename... Others, typename... Excludes>
	entt::view<entt::get_t<Type, Others...>, entt::exclude_t<Excludes...>> getView(entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	template<typename Type>
	entt::entity getEntity(const Type& component) const;

	template<typename Type>
	const Type* getComponent(entt::entity entity) const;

	template<typename Type>
	Type* getComponent(entt::entity entity);

	template<typename Type>
	const Type* getSingleton() const;

	template<typename Type>
	Type* getSingleton();

	template<typename Type>
	bool hasComponent(entt::entity entity) const;

	template<typename Type>
	bool hasSingleton() const;
private:
	World() = default;

	template<typename Type>
	void onCreate(const entt::registry& registry, entt::entity entity);

	template<typename Type>
	void onModify(const entt::registry& registry, entt::entity entity);

	template<typename Type>
	void onDestroy(const entt::registry& registry, entt::entity entity);

	template<typename Event>
	const std::vector<entt::delegate<void(Event&)>>* getDelegates() const;

	template<typename Event>
	std::vector<entt::delegate<void(Event&)>>& getDelegates();

	entt::registry m_registry;
	entt::dense_map<entt::id_type, std::shared_ptr<void>> m_dispatcher;

	friend World& getWorld();
};

World& getWorld();

}

#include "world.inl"
