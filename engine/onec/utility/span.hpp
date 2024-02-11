#pragma once

#include <cstddef>
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
	Span(Type* data, std::ptrdiff_t count);
	Span(Type* first, Type* last);
	Span(std::initializer_list<Type> initializerList) requires std::is_const_v<Type>;

	template<std::size_t Count>
	Span(Type(&array)[Count]);

	template<typename Container> 
	requires std::ranges::contiguous_range<Container> && std::convertible_to<typename Container::value_type*, Type*>
	Span(Container& container);

	Type& operator[](std::ptrdiff_t index) const;
	operator Span<const Type>() const requires (!std::is_const_v<Type>);

	Type* begin() const;
	Type* end() const;

	Type* getData() const;
	std::ptrdiff_t getCount() const;
	std::ptrdiff_t getStride() const;
	std::ptrdiff_t getByteCount() const;
	bool isEmpty() const;
private:
	Type* m_data;
	std::ptrdiff_t m_count;
};

template<typename Type>
auto asBytes(const Span<Type>&& span);

template<typename Type>
auto asBytes(Type* data, std::ptrdiff_t count);

template<typename Type>
auto asBytes(Type* first, Type* last);

template<typename Type>
auto asBytes(std::initializer_list<Type> initializerList);

template<typename Type, std::size_t Count>
auto asBytes(Type(&array)[Count]);

template<typename Container>
requires std::ranges::contiguous_range<Container>
auto asBytes(Container& container);

}

#include "span.inl"
