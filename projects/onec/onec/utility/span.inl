#include "span.hpp"
#include "../config/config.hpp"
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <ranges>

namespace onec
{

template<typename Type>
inline Span<Type>::Span() :
	m_data{ nullptr },
	m_count{ 0 }
{

}

template<typename Type>
inline Span<Type>::Span(Type* const data, const int count) :
	m_data{ data },
	m_count{ count }
{

}

template<typename Type>
inline Span<Type>::Span(Type* const first, Type* const last) :
	m_data{ first },
	m_count{ static_cast<int>(last - first) }
{

}

template<typename Type>
inline Span<Type>::Span(const std::initializer_list<Type> initializerList) requires std::is_const_v<Type> :
    m_data{ std::addressof(*initializerList.begin()) },
    m_count{ static_cast<int>(initializerList.size()) }
{

}

template<typename Type>
template<std::size_t Count>
inline Span<Type>::Span(Type(&array)[Count]) :
	m_data{ array },
	m_count{ static_cast<int>(Count) }
{

}

template<typename Type>
template<typename Container>
requires std::ranges::contiguous_range<Container> && std::convertible_to<typename Container::value_type*, Type*>
Span<Type>::Span(Container& container) :
	m_data{ container.data() },
	m_count{ static_cast<int>(container.size()) }
{

}

template<typename Type> 
inline Span<Type>::operator Span<const Type>() const requires (!std::is_const_v<Type>)
{
	return Span<const Type>{ m_data, m_count };
}

template<typename Type>
inline Type& Span<Type>::operator[](std::size_t index) const
{
	ONEC_ASSERT(index < static_cast<std::size_t>(m_count), "Index must be smaller than count");

	return m_data[index];
}

template<typename Type>
inline Type* Span<Type>::begin() const
{
	return m_data;
}

template<typename Type>
inline Type* Span<Type>::end() const
{
	return m_data + static_cast<std::size_t>(m_count);
}

template<typename Type>
inline Type* Span<Type>::getData() const
{
	return m_data;
}

template<typename Type>
inline int Span<Type>::getCount() const
{
	return m_count;
}

template<typename Type>
inline int Span<Type>::getStride() const
{
	return sizeof(Type);
}

template<typename Type>
inline int Span<Type>::getByteCount() const
{
	return m_count * static_cast<int>(sizeof(Type));
}

template<typename Type>
inline bool Span<Type>::isEmpty() const
{
	return m_count == 0;
}

template<typename Type>
auto asBytes(const Span<Type>&& span)
{
	if constexpr (std::is_const_v<Type>)
	{
		return Span<const std::byte>{ reinterpret_cast<const std::byte*>(span.getData()), span.getByteCount() };
	}
	else
	{
		return Span<std::byte>{ reinterpret_cast<std::byte*>(span.getData()), span.getByteCount() };
	}
}

template<typename Type>
auto asBytes(Type* const data, const int count)
{
	return asBytes(Span<Type>{ data, count });
}

template<typename Type>
auto asBytes(Type* const first, Type* const last)
{
	return asBytes(Span<Type>{ first, last });
}

template<typename Type>
auto asBytes(const std::initializer_list<Type> initializerList)
{
	return asBytes(Span<const Type>{ initializerList });
}

template<typename Type, std::size_t Count>
auto asBytes(Type(&data)[Count])
{
	return asBytes(Span<Type>{ data });
}

template<typename Container>
requires std::ranges::contiguous_range<Container>
auto asBytes(Container& container)
{
	return asBytes(Span<typename Container::value_type>{ container });
}

}
