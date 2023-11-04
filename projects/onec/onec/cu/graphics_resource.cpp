#include "graphics_resource.hpp"
#include "buffer.hpp"
#include "array.hpp"
#include "../config/cu.hpp"
#include "../graphics/buffer.hpp"
#include "../graphics/texture.hpp"
#include "../graphics/renderbuffer.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <utility>

namespace onec
{
namespace cu
{

GraphicsResource::GraphicsResource() :
	m_handle{ cudaGraphicsResource_t{} },
	m_stream{ cudaStream_t{} }
{

}

GraphicsResource::GraphicsResource(onec::Buffer& buffer, const unsigned int flags)
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_handle, buffer.getHandle(), flags));
}

GraphicsResource::GraphicsResource(onec::Texture& texture, const unsigned int flags)
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_handle, texture.getHandle(), texture.getTarget(), flags));
}

GraphicsResource::GraphicsResource(onec::Renderbuffer& renderbuffer, const unsigned int flags)
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_handle, renderbuffer.getHandle(), GL_RENDERBUFFER, flags));
}

GraphicsResource::GraphicsResource(GraphicsResource&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, cudaGraphicsResource_t{}) },
	m_stream{ std::exchange(other.m_stream, cudaStream_t{}) }
{

}

GraphicsResource::~GraphicsResource()
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_handle));
	}
}

GraphicsResource& GraphicsResource::operator=(GraphicsResource&& other) noexcept
{
	if (this != &other)
	{
		if (!isEmpty())
		{
			CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_handle));
		}

		m_handle = std::exchange(other.m_handle, cudaGraphicsResource_t{});
		m_stream = std::exchange(other.m_stream, cudaStream_t{});
	}

	return *this;
}

void GraphicsResource::initialize(onec::Buffer& buffer, const unsigned int flags)
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_handle));
	}

	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_handle, buffer.getHandle(), flags));
}

void GraphicsResource::initialize(onec::Texture& texture, const unsigned int flags)
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_handle));
	}

	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_handle, texture.getHandle(), texture.getTarget(), flags));
}

void GraphicsResource::initialize(onec::Renderbuffer& renderbuffer, const unsigned int flags)
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_handle));
	}

	CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&m_handle, renderbuffer.getHandle(), GL_RENDERBUFFER, flags));
}

void GraphicsResource::release()
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_handle));

		m_handle = cudaGraphicsResource_t{};
		m_stream = cudaStream_t{};
	}
}

void GraphicsResource::map(cudaStream_t stream)
{
	CU_CHECK_ERROR(cudaGraphicsMapResources(1, &m_handle, m_stream));
}

void GraphicsResource::unmap()
{
	CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_handle, m_stream));
}

cudaGraphicsResource_t GraphicsResource::getHandle()
{
	return m_handle;
}

BufferView GraphicsResource::getBufferView() const
{
	void* data;
	size_t count;

	CU_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&data, &count, m_handle));

	return BufferView{ static_cast<std::byte*>(data), static_cast<int>(count) };
}

ArrayView GraphicsResource::getArrayView(const int layer, const int mipLevel) const
{
	cudaArray_t array;
	cudaExtent extent;
	cudaChannelFormatDesc format;
	unsigned int flags;

	CU_CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, m_handle, static_cast<unsigned int>(layer), static_cast<unsigned int>(mipLevel)));
	CU_CHECK_ERROR(cudaArrayGetInfo(&format, &extent, &flags, array));

	return ArrayView{ array, glm::ivec3{ extent.width, extent.height, extent.depth }, format, flags };
}

bool GraphicsResource::isEmpty() const
{
	return m_handle == cudaGraphicsResource_t{};
}

}
}
