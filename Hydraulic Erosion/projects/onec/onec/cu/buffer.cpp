#include "buffer.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include "../graphics/buffer.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <utility>

namespace onec
{
namespace cu
{

Buffer::Buffer() :
	m_data{ nullptr },
	m_count{ 0 },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	
}

Buffer::Buffer(const int count) :
	m_count{ count },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	ONEC_ASSERT(count >= 0, "Count must be greater than or equal to 0");

	if (m_count > 0)
	{
		CU_CHECK_ERROR(cudaMalloc(&m_data, static_cast<size_t>(m_count)));
	}
	else
	{
		m_data = nullptr;
	}
}

Buffer::Buffer(onec::Buffer& buffer, const unsigned int flags) :
	m_data{ nullptr },
	m_count{ 0 },
	m_isMapped{ false }
{
	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, buffer.getHandle(), flags));
}

Buffer::Buffer(const Buffer& other) :
	m_count{ other.m_count },
	m_graphicsResource{ nullptr },
	m_isMapped{ false }
{
	if (!other.isEmpty())
	{
		const size_t count{ static_cast<size_t>(m_count) };
		CU_CHECK_ERROR(cudaMalloc(&m_data, count));
		CU_CHECK_ERROR(cudaMemcpy(m_data, other.m_data, count, cudaMemcpyDeviceToDevice));
	}
	else
	{
		m_data = nullptr;
	}
}

Buffer::Buffer(Buffer&& other) noexcept :
	m_data{ std::exchange(other.m_data, nullptr) },
	m_count{ std::exchange(other.m_count, 0) },
	m_graphicsResource{ std::exchange(other.m_graphicsResource, nullptr) },
	m_isMapped{ std::exchange(other.m_isMapped, false) }
{

}

Buffer::~Buffer()
{
	if (m_graphicsResource != nullptr)
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
	}
	else if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaFree(m_data));
	}
}

Buffer& Buffer::operator=(const Buffer& other)
{
	if (this != &other)
	{
		initialize(other.m_count);

		if (!other.isEmpty())
		{
			CU_CHECK_ERROR(cudaMemcpy(m_data, other.m_data, static_cast<size_t>(m_count), cudaMemcpyDeviceToDevice));
		}
	}

	return *this;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
	if (this != &other)
	{
		if (!isEmpty() && !m_isMapped)
		{
			CU_CHECK_ERROR(cudaFree(m_data));
		}

		m_data = std::exchange(other.m_data, nullptr);
		m_count = std::exchange(other.m_count, 0);
		m_graphicsResource = std::exchange(other.m_graphicsResource, nullptr);
		m_isMapped = std::exchange(other.m_isMapped, false);
	}

	return *this;
}

void Buffer::initialize(const int count)
{
	ONEC_ASSERT(count >= 0, "Count must be greater than or equal to 0");

	if (m_graphicsResource != nullptr)
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
			m_isMapped = false;
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
		m_graphicsResource = nullptr;
	}
	else if (m_count == count)
	{
		return;
	}
	else if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaFree(m_data));
	}

	m_count = count;

	if (m_count > 0)
	{
		CU_CHECK_ERROR(cudaMalloc(&m_data, static_cast<size_t>(m_count)));
	}
	else
	{
		m_data = nullptr;
	}
}

void Buffer::initialize(onec::Buffer& buffer, const unsigned int flags)
{
	release();
	CU_CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&m_graphicsResource, buffer.getHandle(), flags));
}

void Buffer::release()
{
	if (m_graphicsResource != nullptr)
	{
		if (m_isMapped)
		{
			CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
			m_data = nullptr;
			m_count = 0;
			m_isMapped = false;
		}

		CU_CHECK_ERROR(cudaGraphicsUnregisterResource(m_graphicsResource));
		m_graphicsResource = nullptr;
	}
	else if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaFree(m_data));
		m_data = nullptr;
		m_count = 0;
	}
}

void Buffer::upload(const Span<const std::byte>&& data)
{
	CU_CHECK_ERROR(cudaMemcpy(m_data, data.getData(), static_cast<size_t>(m_count), cudaMemcpyHostToDevice));
}

void Buffer::upload(const Span<const std::byte>&& data, const int count)
{
	CU_CHECK_ERROR(cudaMemcpy(m_data, data.getData(), static_cast<size_t>(count), cudaMemcpyHostToDevice));
}

void Buffer::upload(const Span<const std::byte>&& data, const int offset, const int count)
{
	CU_CHECK_ERROR(cudaMemcpy(static_cast<std::byte*>(m_data) + offset, data.getData(), static_cast<size_t>(count), cudaMemcpyHostToDevice));
}

void Buffer::download(const Span<std::byte>&& data) const
{
	CU_CHECK_ERROR(cudaMemcpy(data.getData(), m_data, static_cast<size_t>(m_count), cudaMemcpyDeviceToHost));
}

void Buffer::download(const Span<std::byte>&& data, const int count) const
{
	CU_CHECK_ERROR(cudaMemcpy(data.getData(), m_data, static_cast<size_t>(count), cudaMemcpyDeviceToHost));
}

void Buffer::download(const Span<std::byte>&& data, const int offset, const int count) const
{
	CU_CHECK_ERROR(cudaMemcpy(data.getData(), static_cast<std::byte*>(m_data) + offset, static_cast<size_t>(count), cudaMemcpyDeviceToHost));
}

void Buffer::map()
{
	ONEC_ASSERT(m_graphicsResource != nullptr, "Graphics resource must be registered");
	ONEC_ASSERT(!m_isMapped, "Buffer must be unmapped");

	CU_CHECK_ERROR(cudaGraphicsMapResources(1, &m_graphicsResource));

	size_t count;
	CU_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&m_data, &count, m_graphicsResource));

	m_count = static_cast<int>(count);
	m_isMapped = true;
}

void Buffer::unmap()
{
	ONEC_ASSERT(m_graphicsResource != nullptr, "Graphics resource must be registered");
	ONEC_ASSERT(m_isMapped, "Buffer must be mapped");

	CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));

	m_count = 0;
	m_isMapped = false;
}

int Buffer::getCount() const
{
	return m_count;
}

bool Buffer::isEmpty() const
{
	return m_count == 0;
}

}
}
