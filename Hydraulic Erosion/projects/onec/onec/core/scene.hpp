#pragma once

#include <entt/entt.hpp>
#include <memory>

namespace onec
{

class Scene
{
public:
	Scene(const Scene& other) = delete;
	Scene(Scene&& other) = delete;
	
	~Scene() = default;

	Scene& operator=(const Scene& other) = delete;
	Scene& operator=(Scene&& other) = delete;

	entt::entity addEntity();
	void removeEntity(const entt::entity entity);
	void removeEntities();
	void removeSystems();

	template<typename Event>
	void dispatch(Event&& event = Event{});
	
	template<typename Type, typename... Args>
	decltype(auto) addComponent(const entt::entity entity, Args&&... args);

	template<typename Type, typename... Args>
	decltype(auto) addSingleton(Args&&... args);

	template<typename Event, auto system, typename... Instance>
	void addSystem(Instance&&... instance);

	template<typename Type>
	void removeComponent(const entt::entity entity);

	template<typename Type>
	void removeComponents();

	template<typename Type>
	void removeSingleton();

	template<typename Event, auto system, typename... Instance>
	void removeSystem(Instance&&... instance);

	template<typename Event>
	void removeSystems();

	void setName(const std::string_view& name);

	template<typename Type, typename... Args>
	decltype(auto) setComponent(const entt::entity entity, Args&&... args);

	template<typename Type, typename... Args>
	decltype(auto) setSingleton(Args&&... args);

	const std::string& getName() const;
	entt::entity getSingleton() const;
	int getEntityCount() const;
	bool hasEntities() const;
	bool hasEntity(const entt::entity entity) const;

	template<typename Type>
	entt::entity getEntity(Type& component);

	template<typename Type, typename... Others, typename... Excludes>
	auto getView(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{}) const;

	template<typename Type, typename... Others, typename... Excludes>
	auto getView(const entt::exclude_t<Excludes...> excludes = entt::exclude_t{});

	template<typename Type>
	const Type* getComponent(const entt::entity entity) const;

	template<typename Type>
	Type* getComponent(const entt::entity entity);

	template<typename Type>
	const Type* getSingleton() const;

	template<typename Type>
	Type* getSingleton();

	template<typename Type>
	bool hasComponent(const entt::entity entity) const;

	template<typename Type>
	bool hasSingleton() const;
private:
	Scene(const std::string_view& name = "Scene");

	template<typename Type>
	void onCreate(const entt::registry& registry, const entt::entity entity);

	template<typename Type>
	void onModify(const entt::registry& registry, const entt::entity entity);

	template<typename Type>
	void onDestroy(const entt::registry& registry, const entt::entity entity);

	template<typename Event>
	const auto* getSystems() const;

	template<typename Event>
	auto& getSystems();

	std::string m_name;
	entt::registry m_registry;
	entt::entity m_singleton;
	entt::dense_map<entt::id_type, std::shared_ptr<void>> m_systems;

	friend Scene& getScene();
};

template<typename Type>
struct OnCreate
{
	entt::entity entity;
};

template<typename Type>
struct OnModify
{
	entt::entity entity;
};

template<typename Type>
struct OnDestroy
{
	entt::entity entity;
};

Scene& getScene();

namespace internal
{

template<typename Event>
struct OnCreateTrait
{
	using type = void;
	static inline constexpr bool value{ false };
};

template<typename Type>
struct OnCreateTrait<OnCreate<Type>>
{
	using type = Type;
	static inline constexpr bool value{ true };
};

template<typename Event>
struct OnModifyTrait
{
	using type = void;
	static inline constexpr bool value{ false };
};

template<typename Type>
struct OnModifyTrait<OnModify<Type>>
{
	using type = Type;
	static inline constexpr bool value{ true };
};

template<typename Event>
struct OnDestroyTrait
{
	using type = void;
	static inline constexpr bool value{ false };
};

template<typename Type>
struct OnDestroyTrait<OnDestroy<Type>>
{
	using type = Type;
	static inline constexpr bool value{ true };
};

}
}

#include "scene.inl"
