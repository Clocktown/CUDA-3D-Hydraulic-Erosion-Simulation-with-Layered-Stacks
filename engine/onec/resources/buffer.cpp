#include "buffer.hpp"
#include "graphics_buffer.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>
#include <utility>
#include <type_traits>

namespace onec
{

Buffer::Buffer() :
	m_data{ nullptr },
	m_graphicsResource{},
	m_count{ 0 }
{

}

Buffer::Buffer(const std::ptrdiff_t count)
{
	create(count);
}

Buffer::Buffer(const Span<const std::byte>&& source)
{
	create(std::forward<const Span<const std::byte>>(source));
}

Buffer::Buffer(GraphicsBuffer& source)
{
	create(source);
}

Buffer::Buffer(Buffer&& other) noexcept :
	m_data{ std::exchange(other.m_data, nullptr) },
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

		m_data = std::exchange(other.m_data, nullptr);
		m_graphicsResource = std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{});
		m_count = std::exchange(other.m_count, 0);
	}

	return *this;
}

void Buffer::initialize(const std::ptrdiff_t count)
{
	destroy();
	create(count);
}

void Buffer::initialize(const Span<const std::byte>&& source)
{
	destroy();
	create(std::forward<const Span<const std::byte>>(source));
}

void Buffer::initialize(GraphicsBuffer& source)
{
	destroy();
	create(source);
}

void Buffer::release()
{
	destroy();

	m_data = nullptr;
	m_graphicsResource = cudaGraphicsResource_t{};
	m_count = 0;
}

void Buffer::upload(const Span<const std::byte>&& source)
{
	CU_CHECK_ERROR(cudaMemcpy(m_data, source.getData(), static_cast<size_t>(source.getCount()), cudaMemcpyHostToDevice));
}

void Buffer::upload(const Span<const std::byte>&& source, const std::ptrdiff_t count)
{
	CU_CHECK_ERROR(cudaMemcpy(m_data, source.getData(), static_cast<size_t>(count), cudaMemcpyHostToDevice));
}

void Buffer::upload(const Span<const std::byte>&& source, const std::ptrdiff_t offset, const std::ptrdiff_t count)
{
	CU_CHECK_ERROR(cudaMemcpy(m_data + offset, source.getData(), static_cast<size_t>(count), cudaMemcpyHostToDevice));
}

void Buffer::download(const Span<std::byte>&& destination) const
{
	CU_CHECK_ERROR(cudaMemcpy(destination.getData(), m_data, static_cast<size_t>(destination.getCount()), cudaMemcpyDeviceToHost));
}

void Buffer::download(const Span<std::byte>&& destination, const std::ptrdiff_t count) const
{
	CU_CHECK_ERROR(cudaMemcpy(destination.getData(), m_data, static_cast<size_t>(count), cudaMemcpyDeviceToHost));
}

void Buffer::download(const Span<std::byte>&& destination, const std::ptrdiff_t offset, const std::ptrdiff_t count) const
{
	CU_CHECK_ERROR(cudaMemcpy(destination.getData(), m_data + offset, static_cast<size_t>(count), cudaMemcpyDeviceToHost));
}

const std::byte* Buffer::getData() const
{
	return m_data;
}

std::byte* Buffer::getData()
{
	return m_data;
}

std::ptrdiff_t Buffer::getCount() const
{
	return m_count;
}

bool Buffer::isEmpty() const
{
	return m_count == 0;
}

void Buffer::create(std::ptrdiff_t count)
{
	ONEC_ASSERT(count >= 0, "Count must be greater than zero");

	if (count != 0)
	{
		void* data;
		CU_CHECK_ERROR(cudaMalloc(&data, static_cast<size_t>(count)));

		m_data = static_cast<std::byte*>(data);
	}
	else
	{
		m_data = nullptr;
	}

	m_graphicsResource = cudaGraphicsResource_t{};
	m_count = count;
}

void Buffer::create(const Span<const std::byte>&& source)
{
	if (!source.isEmpty())
	{
		void* data;
		const size_t count{ static_cast<size_t>(source.getCount()) };
		CU_CHECK_ERROR(cudaMalloc(&data, count));
		CU_CHECK_ERROR(cudaMemcpy(data, source.getData(), count, cudaMemcpyHostToDevice));

		m_data = static_cast<std::byte*>(data);
	}
	else
	{
		m_data = nullptr;
	}

	m_graphicsResource = cudaGraphicsResource_t{};
	m_count = source.getCount();
}

void Buffer::create(onec::GraphicsBuffer& source)
{
	if (!source.isEmpty())
	{
		cudaGraphicsResource_t graphicsResource{ source.getGraphicsResource() };
		CU_CHECK_ERROR(cudaGraphicsMapResources(1, &graphicsResource));

		void* data;
		size_t count;
		CU_CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&data, &count, graphicsResource));

		m_data = static_cast<std::byte*>(data);
		m_graphicsResource = graphicsResource;
		m_count = static_cast<int>(count);
	}
	else
	{
		m_data = nullptr;
		m_graphicsResource = cudaGraphicsResource_t{};
		m_count = 0;
	}
}

void Buffer::destroy()
{
	if (m_graphicsResource != cudaGraphicsResource_t{})
	{
		CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
	}
	else if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaFree(m_data));
	}
}

}
