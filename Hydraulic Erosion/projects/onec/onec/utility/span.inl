#include "span.hpp"
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
inline Span<Type>::Span(const std::initializer_list<Type>& data) requires std::is_const_v<Type> :
m_data{ std::addressof(*data.begin()) },
m_count{ static_cast<int>(data.size()) }
{

}

template<typename Type>
template<size_t count>
inline Span<Type>::Span(Type(&data)[count]) :
	m_data{ data },
	m_count{ static_cast<int>(count) }
{

}

template<typename Type>
template<typename Container>
requires std::ranges::contiguous_range<Container> && std::convertible_to<typename Container::value_type*, Type*>
Span<Type>::Span(Container& data) :
	m_data{ data.data() },
	m_count{ static_cast<int>(data.size()) }
{

}

template<typename Type> 
inline Span<Type>::operator Span<const Type>() const
{
	return Span<const Type>{ m_data, m_count };
}

template<typename Type>
inline Type* Span<Type>::begin() const
{
	return m_data;
}

template<typename Type>
inline Type* Span<Type>::end() const
{
	return m_data + static_cast<size_t>(m_count);
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
auto asBytes(const Span<Type>&& data)
{
	if constexpr (std::is_const_v<Type>)
	{
		return Span<const std::byte>{ reinterpret_cast<const std::byte*>(data.getData()), data.getByteCount() };
	}
	else
	{
		return Span<std::byte>{ reinterpret_cast<std::byte*>(data.getData()), data.getByteCount() };
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
auto asBytes(const std::initializer_list<Type>& data)
{
	return asBytes(Span<Type>{ data });
}

template<typename Type, size_t count>
auto asBytes(Type(&data)[count])
{
	return asBytes(Span<Type>{ data });
}

template<typename Container>
requires std::ranges::contiguous_range<Container>
auto asBytes(Container& data)
{
	return asBytes(Span<typename Container::value_type>{ data });
}

}
