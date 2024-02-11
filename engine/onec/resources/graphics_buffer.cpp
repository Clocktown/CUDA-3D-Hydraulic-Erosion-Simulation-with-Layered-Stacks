#include "graphics_buffer.hpp"
#include "../config/gl.hpp"
#include "../config/cu.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <utility>
#include <type_traits>

namespace onec
{

GraphicsBuffer::GraphicsBuffer() :
	m_handle{ GL_NONE },
	m_bindlessHandle{ GL_NONE },
	m_graphicsResource{},
	m_count{ 0 }
{

}

GraphicsBuffer::GraphicsBuffer(const std::ptrdiff_t count, const bool cudaAccess)
{
	create(Span<const std::byte>{ nullptr, count }, cudaAccess);
}

GraphicsBuffer::GraphicsBuffer(const Span<const std::byte>&& source, const bool cudaAccess)
{
	create(std::forward<const Span<const std::byte>>(source), cudaAccess);
}

GraphicsBuffer::GraphicsBuffer(GraphicsBuffer&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_bindlessHandle{ std::exchange(other.m_bindlessHandle, GL_NONE) },
	m_graphicsResource{ std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{}) },
	m_count{ std::exchange(other.m_count, 0) }
{

}

GraphicsBuffer::~GraphicsBuffer()
{
	destroy();
}

GraphicsBuffer& GraphicsBuffer::operator=(GraphicsBuffer&& other) noexcept
{
	if (this != &other)
	{
		destroy();

		m_handle = std::exchange(other.m_handle, GL_NONE);
		m_bindlessHandle = std::exchange(other.m_bindlessHandle, GL_NONE);
		m_graphicsResource = std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{});
		m_count = std::exchange(other.m_count, 0);
	}

	return *this;
}

void GraphicsBuffer::initialize(const std::ptrdiff_t count, const bool cudaAccess)
{
	destroy();
	create(Span<const std::byte>{ nullptr, count }, cudaAccess);
}

void GraphicsBuffer::initialize(const Span<const std::byte>&& source, const bool cudaAccess)
{
	destroy();
	create(std::forward<const Span<const std::byte>>(source), cudaAccess);
}

void GraphicsBuffer::release()
{
	destroy();

	m_handle = GL_NONE;
	m_bindlessHandle = GL_NONE;
	m_graphicsResource = cudaGraphicsResource_t{};
	m_count = 0;
}

void GraphicsBuffer::upload(const Span<const std::byte>&& source)
{
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, 0, source.getCount(), source.getData()));
}

void GraphicsBuffer::upload(const Span<const std::byte>&& source, const std::ptrdiff_t count)
{
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, 0, count, source.getData()));
}

void GraphicsBuffer::upload(const Span<const std::byte>&& source, const std::ptrdiff_t offset, const std::ptrdiff_t count)
{
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, offset, count, source.getData()));
}

void GraphicsBuffer::download(const Span<std::byte>&& destination) const
{
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, 0, destination.getCount(), destination.getData()));
}

void GraphicsBuffer::download(const Span<std::byte>&& destination, const std::ptrdiff_t count) const
{
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, 0, count, destination.getData()));
}

void GraphicsBuffer::download(const Span<std::byte>&& destination, const std::ptrdiff_t offset, const std::ptrdiff_t count) const
{
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, offset, count, destination.getData()));
}

GLuint GraphicsBuffer::getHandle()
{
	return m_handle;
}

GLuint64EXT GraphicsBuffer::getBindlessHandle()
{
	return m_bindlessHandle;
}

cudaGraphicsResource_t GraphicsBuffer::getGraphicsResource()
{
	return m_graphicsResource;
}

std::ptrdiff_t GraphicsBuffer::getCount() const
{
	return m_count;
}

bool GraphicsBuffer::isEmpty() const
{
	return m_count == 0;
}

void GraphicsBuffer::create(const Span<const std::byte>&& source, const bool cudaAccess)
{
	if (!source.isEmpty())
	{
		GLuint handle;
		const std::ptrdiff_t count{ source.getCount() };

		GL_CHECK_ERROR(glCreateBuffers(1, &handle));
		GL_CHECK_ERROR(glNamedBufferStorage(handle, count, source.getData(), GL_DYNAMIC_STORAGE_BIT));

		if (cudaAccess)
		{
			CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, handle, cudaGraphicsMapFlagsNone));
		}
		else
		{
			m_graphicsResource = cudaGraphicsResource_t{};
		}

		GL_CHECK_ERROR(glGetNamedBufferParameterui64vNV(handle, GL_BUFFER_GPU_ADDRESS_NV, &m_bindlessHandle));
		GL_CHECK_ERROR(glMakeNamedBufferResidentNV(handle, GL_READ_WRITE));

		m_handle = handle;
		m_count = count;
	}
	else
	{
		m_handle = GL_NONE;
		m_bindlessHandle = GL_NONE;
		m_graphicsResource = cudaGraphicsResource_t{};
		m_count = 0;
	}
}

void GraphicsBuffer::destroy()
{
	if (!isEmpty())
	{
		const cudaGraphicsResource_t graphicsResource{ m_graphicsResource };

		if (graphicsResource != cudaGraphicsResource_t{})
		{
			CU_CHECK_ERROR(cudaGraphicsUnregisterResource(graphicsResource));
		}

		GL_CHECK_ERROR(glDeleteBuffers(1, &m_handle));
	}
}

}
