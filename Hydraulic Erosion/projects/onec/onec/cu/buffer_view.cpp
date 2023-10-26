#include "buffer_view.hpp"
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

BufferView::BufferView() :
	m_data{ nullptr },
	m_count{ 0 }
{
}

BufferView::BufferView(Buffer& buffer) :
	m_data{ buffer.getData() },
	m_count{ buffer.getCount() }
{

}

BufferView::BufferView(Buffer& buffer, const int count) :
	m_data{ buffer.getData() },
	m_count{ count }
{

}

BufferView::BufferView(std::byte* const data, const int count) :
	m_data{ data },
	m_count{ count }
{

}

BufferView& BufferView::operator=(Buffer& buffer)
{
	m_data = buffer.getData();
	m_count = buffer.getCount();

	return *this;
}

void BufferView::upload(const Span<const std::byte>&& data)
{
	CU_CHECK_ERROR(cudaMemcpy(m_data, data.getData(), static_cast<size_t>(m_count), cudaMemcpyHostToDevice));
}

void BufferView::upload(const Span<const std::byte>&& data, const int count)
{
	CU_CHECK_ERROR(cudaMemcpy(m_data, data.getData(), static_cast<size_t>(count), cudaMemcpyHostToDevice));
}

void BufferView::upload(const Span<const std::byte>&& data, const int offset, const int count)
{
	CU_CHECK_ERROR(cudaMemcpy(m_data + offset, data.getData(), static_cast<size_t>(count), cudaMemcpyHostToDevice));
}

void BufferView::download(const Span<std::byte>&& data) const
{
	CU_CHECK_ERROR(cudaMemcpy(data.getData(), m_data, static_cast<size_t>(m_count), cudaMemcpyDeviceToHost));
}

void BufferView::download(const Span<std::byte>&& data, const int count) const
{
	CU_CHECK_ERROR(cudaMemcpy(data.getData(), m_data, static_cast<size_t>(count), cudaMemcpyDeviceToHost));
}

void BufferView::download(const Span<std::byte>&& data, const int offset, const int count) const
{
	CU_CHECK_ERROR(cudaMemcpy(data.getData(), m_data + offset, static_cast<size_t>(count), cudaMemcpyDeviceToHost));
}

const std::byte* BufferView::getData() const
{
	return m_data;
}

std::byte* BufferView::getData()
{
	return m_data;
}

int BufferView::getCount() const
{
	return m_count;
}

bool BufferView::isEmpty() const
{
	return m_count == 0;
}

}
}
