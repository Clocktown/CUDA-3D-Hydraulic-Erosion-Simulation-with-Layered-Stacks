#include "buffer.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>
#include <utility>

namespace onec
{
namespace cu
{

Buffer::Buffer() :
	m_data{ nullptr },
	m_count{ 0 }
{
	
}

Buffer::Buffer(const int count) :
	m_count{ count }
{
	ONEC_ASSERT(count >= 0, "Count must be greater than or equal to 0");

	if (m_count > 0)
	{
		CU_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&m_data), static_cast<size_t>(m_count)));
	}
	else
	{
		m_data = nullptr;
	}
}

Buffer::Buffer(const Buffer& other) :
	m_count{ other.m_count }
{
	if (!other.isEmpty())
	{
		const size_t count{ static_cast<size_t>(m_count) };
		CU_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&m_data), count));
		CU_CHECK_ERROR(cudaMemcpy(m_data, other.m_data, count, cudaMemcpyDeviceToDevice));
	}
	else
	{
		m_data = nullptr;
	}
}

Buffer::Buffer(Buffer&& other) noexcept :
	m_data{ std::exchange(other.m_data, nullptr) },
	m_count{ std::exchange(other.m_count, 0) }
{

}

Buffer::~Buffer()
{
	if (!isEmpty())
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
		if (!isEmpty())
		{
			CU_CHECK_ERROR(cudaFree(m_data));
		}

		m_data = std::exchange(other.m_data, nullptr);
		m_count = std::exchange(other.m_count, 0);
	}

	return *this;
}

void Buffer::initialize(const int count)
{
	ONEC_ASSERT(count >= 0, "Count must be greater than or equal to 0");
	
	if (m_count == count)
	{
		return;
	}

	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaFree(m_data));
	}

	m_count = count;

	if (m_count > 0)
	{
		CU_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&m_data), static_cast<size_t>(m_count)));
	}
	else
	{
		m_data = nullptr;
	}
}

void Buffer::release()
{
	if (!isEmpty())
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
	CU_CHECK_ERROR(cudaMemcpy(m_data + offset, data.getData(), static_cast<size_t>(count), cudaMemcpyHostToDevice));
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
	CU_CHECK_ERROR(cudaMemcpy(data.getData(), m_data + offset, static_cast<size_t>(count), cudaMemcpyDeviceToHost));
}

const std::byte* Buffer::getData() const
{
	return m_data;
}

std::byte* Buffer::getData()
{
	return m_data;
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
