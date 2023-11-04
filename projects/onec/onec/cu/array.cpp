#include "array.hpp"
#include "format.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include "../utility/io.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <filesystem>
#include <type_traits>

namespace onec
{
namespace cu
{

Array::Array() :
	m_handle{ cudaArray_t{} },
	m_size{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault }
{

}

Array::Array(const glm::ivec3& size, const cudaChannelFormatDesc& format, const unsigned int flags) :
	m_size{ size },
	m_format{ format },
	m_flags{ flags }
{
	ONEC_ASSERT(size.x >= 0, "Size x must be greater than or equal to 0");
	ONEC_ASSERT(size.y >= 0, "Size y must be greater than or equal to 0");
	ONEC_ASSERT(size.z >= 0, "Size z must be greater than or equal to 0");

	if (m_size != glm::ivec3{ 0 })
	{
		const cudaExtent extent{ static_cast<size_t>(m_size.x), static_cast<size_t>(m_size.y), static_cast<size_t>(m_size.z) };
		CU_CHECK_ERROR(cudaMalloc3DArray(&m_handle, &m_format, extent, m_flags));
	}
	else
	{
		m_handle = cudaArray_t{};
	}
}

Array::Array(const std::filesystem::path& file, const unsigned int flags) :
	m_flags{ flags }
{
	m_size.z = 0;
	const auto data{ readImage(file, reinterpret_cast<glm::ivec2&>(m_size), m_format) };

	const cudaExtent extent{ static_cast<size_t>(m_size.x), static_cast<size_t>(m_size.y), 0 };
	CU_CHECK_ERROR(cudaMalloc3DArray(&m_handle, &m_format, extent, m_flags));

	upload({ data.get(), m_size.x * m_size.y });
}

Array::Array(const Array& other) :
	m_size{ other.m_size },
	m_format{ other.m_format },
	m_flags{ other.m_flags }
{
	if (!other.isEmpty())
	{
		const cudaExtent extent{ static_cast<size_t>(m_size.x), static_cast<size_t>(m_size.y), static_cast<size_t>(m_size.z) };
		CU_CHECK_ERROR(cudaMalloc3DArray(&m_handle, &m_format, extent, m_flags));

		const cudaMemcpy3DParms copyParms{ .srcArray{ other.m_handle },
								           .dstArray{ m_handle },
								           .extent{ static_cast<size_t>(m_size.x), static_cast<size_t>(glm::max(m_size.y, 1)), static_cast<size_t>(glm::max(m_size.z, 1)) },
								           .kind{ cudaMemcpyDeviceToDevice } };

		CU_CHECK_ERROR(cudaMemcpy3D(&copyParms));
	}
	else
	{
		m_handle = nullptr;
	}
}

Array::Array(Array&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, cudaArray_t{}) },
	m_size{ std::exchange(other.m_size, glm::ivec3{ 0 }) },
	m_format{ std::exchange(other.m_format, cudaCreateChannelDesc<cudaChannelFormatKindNone>()) },
	m_flags{ std::exchange(other.m_flags, cudaArrayDefault) }
{

}

Array::~Array()
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaFreeArray(m_handle));
	}
}

Array& Array::operator=(const Array& other)
{
	if (this != &other)
	{
		initialize(other.m_size, other.m_format, other.m_flags);

		if (!other.isEmpty())
		{
			const cudaMemcpy3DParms copyParms{ .srcArray{ other.m_handle },
											   .dstArray{ m_handle },
											   .extent{ static_cast<size_t>(m_size.x), static_cast<size_t>(glm::max(m_size.y, 1)), static_cast<size_t>(glm::max(m_size.z, 1)) },
											   .kind{ cudaMemcpyDeviceToDevice } };

			CU_CHECK_ERROR(cudaMemcpy3D(&copyParms));
		}
	}

	return *this;
}

Array& Array::operator=(Array&& other) noexcept
{
	if (this != &other)
	{
		if (!isEmpty())
		{
			CU_CHECK_ERROR(cudaFreeArray(m_handle));
		}

		m_handle = std::exchange(other.m_handle, cudaArray_t{});
		m_size = std::exchange(other.m_size, glm::ivec3{ 0 });
		m_format = std::exchange(other.m_format, cudaCreateChannelDesc<cudaChannelFormatKindNone>());
		m_flags = std::exchange(other.m_flags, cudaArrayDefault);
	}

	return *this;
}

void Array::initialize(const glm::ivec3& size, const cudaChannelFormatDesc& format, const unsigned int flags)
{
	ONEC_ASSERT(size.x >= 0, "Size x must be greater than or equal to 0");
	ONEC_ASSERT(size.y >= 0, "Size y must be greater than or equal to 0");
	ONEC_ASSERT(size.z >= 0, "Size z must be greater than or equal to 0");
	
	if (m_size == size && m_format == format && m_flags == flags)
	{
		return;
	}

	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaFreeArray(m_handle));
	}

	m_size = size;
	m_format = format;
	m_flags = flags;

	if (m_size != glm::ivec3{ 0 })
	{
		const cudaExtent extent{ static_cast<size_t>(m_size.x), static_cast<size_t>(m_size.y), static_cast<size_t>(m_size.z) };
		CU_CHECK_ERROR(cudaMalloc3DArray(&m_handle, &m_format, extent, m_flags));
	}
	else
	{
		m_handle = cudaArray_t{};
	}
}

void Array::release()
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaFreeArray(m_handle));

		m_handle = cudaArray_t{};
		m_size = glm::ivec3{ 0 };
		m_flags = 0;
	}
}

void Array::upload(const Span<const std::byte>&& data)
{
	const glm::ivec3 size{ m_size.x, glm::max(m_size.y, 1), glm::max(m_size.z, 1) };
	upload(std::forward<const Span<const std::byte>&&>(data), glm::ivec3{ 0 }, size);
}

void Array::upload(const Span<const std::byte>&& data, const glm::ivec3& size)
{
	upload(std::forward<const Span<const std::byte>&&>(data), glm::ivec3{ 0 }, size);
}

void Array::upload(const Span<const std::byte>&& data, const glm::ivec3& offset, const glm::ivec3& size)
{
	ONEC_ASSERT(offset.x >= 0, "Offset x must be greater than or equal to 0");
	ONEC_ASSERT(offset.y >= 0, "Offset y must be greater than or equal to 0");
	ONEC_ASSERT(offset.z >= 0, "Offset z must be greater than or equal to 0");
	ONEC_ASSERT(size.x >= 0, "Size x must be greater than or equal to 0");
	ONEC_ASSERT(size.y >= 0, "Size y must be greater than or equal to 0");
	ONEC_ASSERT(size.z >= 0, "Size z must be greater than or equal to 0");

	const cudaExtent extent{ static_cast<size_t>(size.x), static_cast<size_t>(size.y), static_cast<size_t>(size.z) };
	const int stride{ m_format.x + m_format.y + m_format.z + m_format.w };

	const cudaMemcpy3DParms copyParms{ .srcPtr{ make_cudaPitchedPtr(const_cast<std::byte*>(data.getData()), extent.width * static_cast<size_t>(stride), extent.width, extent.height)},
									   .dstArray{ m_handle },
									   .dstPos{ static_cast<size_t>(offset.x), static_cast<size_t>(offset.y), static_cast<size_t>(offset.z) },
									   .extent{ extent },
									   .kind{ cudaMemcpyHostToDevice } };

	CU_CHECK_ERROR(cudaMemcpy3D(&copyParms));
}

void Array::download(const Span<std::byte>&& data) const
{
	const glm::ivec3 size{ m_size.x, glm::max(m_size.y, 1), glm::max(m_size.z, 1) };
	download(std::forward<const Span<std::byte>&&>(data), glm::ivec3{ 0 }, size);
}

void Array::download(const Span<std::byte>&& data, const glm::ivec3& size) const
{
	download(std::forward<const Span<std::byte>&&>(data), glm::ivec3{ 0 }, size);
}

void Array::download(const Span<std::byte>&& data, const glm::ivec3& offset, const glm::ivec3& size) const
{
	ONEC_ASSERT(offset.x >= 0, "Offset x must be greater than or equal to 0");
	ONEC_ASSERT(offset.y >= 0, "Offset y must be greater than or equal to 0");
	ONEC_ASSERT(offset.z >= 0, "Offset z must be greater than or equal to 0");
	ONEC_ASSERT(size.x >= 0, "Size x must be greater than or equal to 0");
	ONEC_ASSERT(size.y >= 0, "Size y must be greater than or equal to 0");
	ONEC_ASSERT(size.z >= 0, "Size z must be greater than or equal to 0");

	const cudaExtent extent{ static_cast<size_t>(size.x), static_cast<size_t>(size.y), static_cast<size_t>(size.z) };
	const int stride{ m_format.x + m_format.y + m_format.z + m_format.w };

	const cudaMemcpy3DParms copyParms{ .srcArray{ m_handle },
									   .srcPos{ static_cast<size_t>(offset.x), static_cast<size_t>(offset.y), static_cast<size_t>(offset.z) },
									   .dstPtr{ make_cudaPitchedPtr(data.getData(), extent.width * static_cast<size_t>(stride), extent.width, extent.height) },
									   .extent{ extent },
									   .kind{ cudaMemcpyDeviceToHost } };

	CU_CHECK_ERROR(cudaMemcpy3D(&copyParms));
}

cudaArray_const_t Array::getHandle() const
{
	return m_handle;
}

cudaArray_t Array::getHandle()
{
	return m_handle;
}

const glm::ivec3& Array::getSize() const
{
	return m_size;
}

const cudaChannelFormatDesc& Array::getFormat() const
{
	return m_format;
}

unsigned int Array::getFlags() const
{
	return m_flags;
}

bool Array::isEmpty() const
{
	return m_handle == cudaArray_t{};
}

}
}
