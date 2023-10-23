#pragma once

#include <initializer_list>
#include <type_traits>
#include <concepts>
#include <ranges>

namespace onec
{

template<typename Type>
class Span
{
public:
	Span();
	Span(Type* const data, const int count);
	Span(Type* const first, Type* const last);
	Span(const std::initializer_list<Type>& data) requires std::is_const_v<Type>;

	template<size_t count>
	Span(Type(&data)[count]);

	template<typename Container> 
	requires std::ranges::contiguous_range<Container> && std::convertible_to<typename Container::value_type*, Type*>
	Span(Container& data);

	operator Span<const Type>() const;

	Type* begin() const;
	Type* end() const;

	Type* getData() const;
	int getCount() const;
	int getStride() const;
	int getByteCount() const;
	bool isEmpty() const;
private:
	Type* m_data;
	int m_count;
};

template<typename Type>
auto asBytes(const Span<Type>&& data);

template<typename Type>
auto asBytes(Type* const data, const int count);

template<typename Type>
auto asBytes(Type* const first, Type* const last);

template<typename Type>
auto asBytes(const std::initializer_list<Type>& data);

template<typename Type, size_t count>
auto asBytes(Type(&data)[count]);

template<typename Container>
requires std::ranges::contiguous_range<Container>
auto asBytes(Container& data);

}

#include "span.inl"
