#pragma once

#include <entt/entt.hpp>

namespace onec 
{

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
