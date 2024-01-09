#include "buffer.hpp"
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

Buffer::Buffer() :
	m_handle{ GL_NONE },
	m_bindlessHandle{ GL_NONE },
	m_graphicsResource{},
	m_count{ 0 }
{

}

Buffer::Buffer(const int count, const bool createBindlessHandle, const bool createGraphicsResource)
{
	create(Span<const std::byte>{ nullptr, count }, createBindlessHandle, createGraphicsResource);
}

Buffer::Buffer(const Span<const std::byte>&& source, const bool createBindlessHandle, const bool createGraphicsResource)
{
	create(std::forward<const Span<const std::byte>>(source), createBindlessHandle, createGraphicsResource);
}

Buffer::Buffer(Buffer&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_bindlessHandle{ std::exchange(other.m_bindlessHandle, GL_NONE) },
	m_graphicsResource{ std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{}) },
	m_count{ std::exchange(other.m_count, 0) }
{

}

Buffer::~Buffer()
{
	destroy();
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
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

void Buffer::initialize(const int count, const bool createBindlessHandle, const bool createGraphicsResource)
{
	destroy();
	create(Span<const std::byte>{ nullptr, count }, createBindlessHandle, createGraphicsResource);
}

void Buffer::initialize(const Span<const std::byte>&& source, const bool createBindlessHandle, const bool createGraphicsResource)
{
	destroy();
	create(std::forward<const Span<const std::byte>>(source), createBindlessHandle, createGraphicsResource);
}

void Buffer::release()
{
	destroy();

	m_handle = GL_NONE;
	m_bindlessHandle = GL_NONE;
	m_graphicsResource = cudaGraphicsResource_t{};
	m_count = 0;
}

void Buffer::upload(const Span<const std::byte>&& source)
{
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, 0, source.getCount(), source.getData()));
}

void Buffer::upload(const Span<const std::byte>&& source, const int count)
{
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, 0, count, source.getData()));
}

void Buffer::upload(const Span<const std::byte>&& source, const int offset, const int count)
{
	GL_CHECK_ERROR(glNamedBufferSubData(m_handle, offset, count, source.getData()));
}

void Buffer::download(const Span<std::byte>&& destination) const
{
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, 0, destination.getCount(), destination.getData()));
}

void Buffer::download(const Span<std::byte>&& destination, const int count) const
{
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, 0, count, destination.getData()));
}

void Buffer::download(const Span<std::byte>&& destination, const int offset, const int count) const
{
	GL_CHECK_ERROR(glGetNamedBufferSubData(m_handle, offset, count, destination.getData()));
}

GLuint Buffer::getHandle()
{
	return m_handle;
}

GLuint64EXT Buffer::getBindlessHandle()
{
	return m_bindlessHandle;
}

cudaGraphicsResource_t Buffer::getGraphicsResource()
{
	return m_graphicsResource;
}

int Buffer::getCount() const
{
	return m_count;
}

bool Buffer::isEmpty() const
{
	return m_count == 0;
}

void Buffer::create(const Span<const std::byte>&& source, const bool createBindlessHandle, const bool createGraphicsResource)
{
	if (!source.isEmpty())
	{
		GLuint handle;
		const int count{ source.getCount() };
		GL_CHECK_ERROR(glCreateBuffers(1, &handle));
		GL_CHECK_ERROR(glNamedBufferStorage(handle, count, source.getData(), GL_DYNAMIC_STORAGE_BIT));

		m_handle = handle;
		m_count = count;

		if (createGraphicsResource)
		{
			cudaGraphicsResource_t graphicsResource;
			CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&graphicsResource, handle, cudaGraphicsMapFlagsNone));

			m_graphicsResource = graphicsResource;
		}
		else
		{
			m_graphicsResource = cudaGraphicsResource_t{};
		}

		if (createBindlessHandle)
		{	
			GL_CHECK_ERROR(glGetNamedBufferParameterui64vNV(handle, GL_BUFFER_GPU_ADDRESS_NV, &m_bindlessHandle));
			GL_CHECK_ERROR(glMakeNamedBufferResidentNV(handle, GL_READ_WRITE));
		}
		else
		{
			m_bindlessHandle = GL_NONE;
		}
	}
	else
	{
		m_handle = GL_NONE;
		m_bindlessHandle = GL_NONE;
		m_graphicsResource = cudaGraphicsResource_t{};
		m_count = 0;
	}
}

void Buffer::destroy()
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
