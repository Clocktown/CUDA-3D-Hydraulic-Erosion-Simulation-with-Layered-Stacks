#include "array_view.hpp"
#include "array.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <type_traits>

namespace onec
{
namespace cu
{

ArrayView::ArrayView() :
	m_handle{ cudaArray_t{} },
	m_size{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault }
{

}

ArrayView::ArrayView(Array& array) :
	m_handle{ array.getHandle() },
	m_size{ array.getSize() },
	m_format{ array.getFormat() },
	m_flags{ array.getFlags() }
{

}

ArrayView::ArrayView(const cudaArray_t array, const glm::ivec3& size, const cudaChannelFormatDesc& format, const unsigned int flags) :
	m_handle{ array },
	m_size{ size },
	m_format{ format },
	m_flags{ flags }
{

}

ArrayView& ArrayView::operator=(Array& array)
{
	m_handle = array.getHandle();
	m_size = array.getSize();
	m_format = array.getFormat();
	m_flags = array.getFlags();

	return *this;
}

void ArrayView::upload(const Span<const std::byte>&& data)
{
	const glm::ivec3 size{ m_size.x, glm::max(m_size.y, 1), glm::max(m_size.z, 1) };
	upload(std::forward<const Span<const std::byte>&&>(data), glm::ivec3{ 0 }, size);
}

void ArrayView::upload(const Span<const std::byte>&& data, const glm::ivec3& size)
{
	upload(std::forward<const Span<const std::byte>&&>(data), glm::ivec3{ 0 }, size);
}

void ArrayView::upload(const Span<const std::byte>&& data, const glm::ivec3& offset, const glm::ivec3& size)
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

void ArrayView::download(const Span<std::byte>&& data) const
{
	const glm::ivec3 size{ m_size.x, glm::max(m_size.y, 1), glm::max(m_size.z, 1) };
	download(std::forward<const Span<std::byte>&&>(data), glm::ivec3{ 0 }, size);
}

void ArrayView::download(const Span<std::byte>&& data, const glm::ivec3& size) const
{
	download(std::forward<const Span<std::byte>&&>(data), glm::ivec3{ 0 }, size);
}

void ArrayView::download(const Span<std::byte>&& data, const glm::ivec3& offset, const glm::ivec3& size) const
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

cudaArray_const_t ArrayView::getHandle() const
{
	return m_handle;
}

cudaArray_t ArrayView::getHandle()
{
	return m_handle;
}

const glm::ivec3& ArrayView::getSize() const
{
	return m_size;
}

const cudaChannelFormatDesc& ArrayView::getFormat() const
{
	return m_format;
}

unsigned int ArrayView::getFlags() const
{
	return m_flags;
}

bool ArrayView::isEmpty() const
{
	return m_handle == cudaArray_t{};
}

}
}
