#include "array.hpp"
#include "format.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include "../graphics/texture.hpp"
#include "../graphics/renderbuffer.hpp"
#include "../utility/io.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <filesystem>
#include <string>
#include <type_traits>

namespace onec
{
namespace cu
{

Array::Array() :
	m_handle{ cudaArray_t{} },
	m_size{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault },
	m_graphicsResource{ cudaGraphicsResource_t{} },
	m_isMapped{ false }
{

}

Array::Array(const glm::ivec3& size, const cudaChannelFormatDesc& format, const unsigned int flags) :
	m_size{ size },
	m_format{ format },
	m_flags{ flags },
	m_graphicsResource{ cudaGraphicsResource_t{} },
	m_isMapped{ false }
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
	m_flags{ flags },
	m_graphicsResource{ cudaGraphicsResource_t{} },
	m_isMapped{ false }
{
	m_size.z = 0;
	const auto data{ readImage(file, reinterpret_cast<glm::ivec2&>(m_size), m_format) };

	const cudaExtent extent{ static_cast<size_t>(m_size.x), static_cast<size_t>(m_size.y), 0 };
	CU_CHECK_ERROR(cudaMalloc3DArray(&m_handle, &m_format, extent, m_flags));

	upload({ data.get(), m_size.x* m_size.y });
}

Array::Array(onec::Texture& texture, const unsigned int flags) :
	m_handle{ cudaArray_t{} },
	m_size{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault },
	m_isMapped{ false }
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, texture.getHandle(), texture.getTarget(), flags));
}

Array::Array(onec::Renderbuffer& renderbuffer, const unsigned int flags) :
	m_handle{ cudaArray_t{} },
	m_size{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault },
	m_isMapped{ false }
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, renderbuffer.getHandle(), GL_RENDERBUFFER, flags));
}

Array::Array(const Array& other) :
	m_size{ other.m_size },
	m_format{ other.m_format },
	m_flags{ other.m_flags },
	m_graphicsResource{ cudaGraphicsResource_t{} },
	m_isMapped{ false }
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
	m_flags{ std::exchange(other.m_flags, cudaArrayDefault) },
	m_graphicsResource{ std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{}) },
	m_isMapped{ std::exchange(other.m_isMapped, false) }
{

}

Array::~Array()
{
	if (m_graphicsResource != cudaGraphicsResource_t{})
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
	}
	else if (!isEmpty())
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
		if (!isEmpty() && !m_isMapped)
		{
			CU_CHECK_ERROR(cudaFreeArray(m_handle));
		}

		m_handle = std::exchange(other.m_handle, cudaArray_t{});
		m_size = std::exchange(other.m_size, glm::ivec3{ 0 });
		m_format = std::exchange(other.m_format, cudaCreateChannelDesc<cudaChannelFormatKindNone>());
		m_flags = std::exchange(other.m_flags, cudaArrayDefault);
		m_graphicsResource = std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{});
		m_isMapped = std::exchange(other.m_isMapped, false);
	}

	return *this;
}

void Array::initialize(const glm::ivec3& size, const cudaChannelFormatDesc& format, const unsigned int flags)
{
	ONEC_ASSERT(size.x >= 0, "Size x must be greater than or equal to 0");
	ONEC_ASSERT(size.y >= 0, "Size y must be greater than or equal to 0");
	ONEC_ASSERT(size.z >= 0, "Size z must be greater than or equal to 0");

	if (m_graphicsResource != cudaGraphicsResource_t{})
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
			m_isMapped = false;
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
		m_graphicsResource = cudaGraphicsResource_t{};
	}
	else if (m_size == size && m_format == format && m_flags == flags)
	{
		return;
	}
	else if (!isEmpty())
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

void Array::initialize(onec::Texture& texture, const unsigned int flags)
{
	release();
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, texture.getHandle(), texture.getTarget(), flags));
}

void Array::initialize(onec::Renderbuffer& renderbuffer, const unsigned int flags)
{
	release();
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_graphicsResource, renderbuffer.getHandle(), GL_RENDERBUFFER, flags));
}

void Array::release()
{
	if (m_graphicsResource != cudaGraphicsResource_t{})
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
			m_handle = cudaArray_t{};
			m_size = glm::ivec3{ 0 };
			m_flags = 0;
			m_isMapped = false;
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
		m_graphicsResource = cudaGraphicsResource_t{};
	}
	else if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaFreeArray(m_handle));
		m_handle = cudaArray_t{};
		m_size = glm::ivec3{ 0 };
		m_flags = 0;
		m_isMapped = false;
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

	const cudaPos pos{ static_cast<size_t>(offset.x), static_cast<size_t>(offset.y), static_cast<size_t>(offset.z) };
	const cudaExtent extent{ static_cast<size_t>(size.x), static_cast<size_t>(size.y), static_cast<size_t>(size.z) };
	const size_t stride{ static_cast<size_t>(m_format.x) + static_cast<size_t>(m_format.y) + static_cast<size_t>(m_format.z) + static_cast<size_t>(m_format.w) };
	const cudaMemcpy3DParms copyParms{ .srcPtr{ make_cudaPitchedPtr(const_cast<std::byte*>(data.getData()), extent.width * stride, extent.width, extent.height)},
									   .dstArray{ m_handle },
									   .dstPos{ pos },
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

	const cudaPos pos{ static_cast<size_t>(offset.x), static_cast<size_t>(offset.y), static_cast<size_t>(offset.z) };
	const cudaExtent extent{ static_cast<size_t>(size.x), static_cast<size_t>(size.y), static_cast<size_t>(size.z) };
	const size_t stride{ static_cast<size_t>(m_format.x) + static_cast<size_t>(m_format.y) + static_cast<size_t>(m_format.z) + static_cast<size_t>(m_format.w) };
	const cudaMemcpy3DParms copyParms{ .srcArray{ m_handle },
									   .srcPos{ pos },
									   .dstPtr{ make_cudaPitchedPtr(data.getData(), extent.width * stride, extent.width, extent.height) },
									   .extent{ extent },
									   .kind{ cudaMemcpyDeviceToHost } };

	CU_CHECK_ERROR(cudaMemcpy3D(&copyParms));
}

void Array::map(const int layer, const int mipLevel)
{
	ONEC_ASSERT(m_graphicsResource != cudaGraphicsResource_t{}, "Graphics resource must be registered");
	ONEC_ASSERT(!m_isMapped, "Buffer must be unmapped");

	CU_CHECK_ERROR(cudaGraphicsMapResources(1, &m_graphicsResource));
	CU_CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&m_handle, m_graphicsResource, static_cast<unsigned int>(layer), static_cast<unsigned int>(mipLevel)));

	cudaExtent extent;
	CU_CHECK_ERROR(cudaArrayGetInfo(&m_format, &extent, &m_flags, m_handle));
	m_size.x = static_cast<int>(extent.width);
	m_size.y = static_cast<int>(extent.height);
	m_size.z = static_cast<int>(extent.depth);
	m_isMapped = true;
}

void Array::unmap()
{
	ONEC_ASSERT(m_graphicsResource != cudaGraphicsResource_t{}, "Graphics resource must be registered");
	ONEC_ASSERT(m_isMapped, "Array must be mapped");

	CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
	m_isMapped = false;
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
