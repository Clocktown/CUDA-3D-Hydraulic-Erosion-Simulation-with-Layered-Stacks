#include "array.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include "../graphics/texture.hpp"
#include "../utility/io.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <utility>
#include <filesystem>
#include <type_traits>

namespace onec
{
namespace cu
{

Array::Array() :
	m_handle{},
	m_textureObject{},
	m_surfaceObject{},
	m_graphicsResource{},
	m_size{ 0 },
	m_format{ cudaCreateChannelDesc<cudaChannelFormatKindNone>() },
	m_flags{ cudaArrayDefault }
{

}

Array::Array(const glm::ivec3 size, const cudaChannelFormatDesc& format, const unsigned int flags, const cudaTextureDesc* const textureDescriptor, const bool createSurfaceObject)
{
	create(size, format, flags, textureDescriptor, createSurfaceObject);
}

Array::Array(const std::filesystem::path& file, const unsigned int flags, const cudaTextureDesc* const textureDescriptor, const bool createSurfaceObject)
{
	create(file, flags, textureDescriptor, createSurfaceObject);
}

Array::Array(onec::Texture& texture, const cudaTextureDesc* const textureDescriptor, const bool createSurfaceObject)
{
	create(texture, textureDescriptor, createSurfaceObject);
}

Array::Array(Array&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, cudaArray_t{}) },
	m_textureObject{ std::exchange(other.m_textureObject, cudaTextureObject_t{}) },
	m_surfaceObject{ std::exchange(other.m_surfaceObject, cudaSurfaceObject_t{}) },
	m_graphicsResource{ std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{}) },
	m_size{ std::exchange(other.m_size, glm::ivec3{ 0 }) },
	m_format{ std::exchange(other.m_format, cudaCreateChannelDesc<cudaChannelFormatKindNone>()) },
	m_flags{ std::exchange(other.m_flags, cudaArrayDefault) }
{

}

Array::~Array()
{
	destroy();
}

Array& Array::operator=(Array&& other) noexcept
{
	if (this != &other)
	{
		destroy();

		m_handle = std::exchange(other.m_handle, cudaArray_t{});
		m_textureObject = std::exchange(other.m_textureObject, cudaTextureObject_t{});
		m_surfaceObject = std::exchange(other.m_surfaceObject, cudaSurfaceObject_t{});
		m_graphicsResource = std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{});
		m_size = std::exchange(other.m_size, glm::ivec3{ 0 });
		m_format = std::exchange(other.m_format, cudaCreateChannelDesc<cudaChannelFormatKindNone>());
		m_flags = std::exchange(other.m_flags, cudaArrayDefault);
	}

	return *this;
}

void Array::initialize(const glm::ivec3 size, const cudaChannelFormatDesc& format, const unsigned int flags, const cudaTextureDesc* textureDescriptor, const bool createSurfaceObject)
{
	destroy();
	create(size, format, flags, textureDescriptor, createSurfaceObject);
}

void Array::initialize(const std::filesystem::path& file, const unsigned int flags, const cudaTextureDesc* const textureDescriptor, const bool createSurfaceObject)
{
	destroy();
	create(file, flags, textureDescriptor, createSurfaceObject);
}

void Array::initialize(onec::Texture& texture, const cudaTextureDesc* const textureDescriptor, const  bool createSurfaceObject)
{
	destroy();
	create(texture, textureDescriptor, createSurfaceObject);
}

void Array::release()
{
	destroy();

	m_handle = cudaArray_t{};
	m_textureObject = cudaTextureObject_t{};
	m_surfaceObject = cudaSurfaceObject_t{};
	m_graphicsResource = cudaGraphicsResource_t{};
	m_size = glm::ivec3{ 0 };
	m_format = cudaCreateChannelDesc<cudaChannelFormatKindNone>();
	m_flags = 0;
}

void Array::upload(const Span<const std::byte>&& source, const glm::ivec3 size)
{
	upload(std::forward<const Span<const std::byte>&&>(source), glm::ivec3{ 0 }, size);
}

void Array::upload(const Span<const std::byte>&& source, const glm::ivec3 offset, const glm::ivec3 size)
{
	ONEC_ASSERT(offset.x >= 0, "Offset x must be greater than or equal to 0");
	ONEC_ASSERT(offset.y >= 0, "Offset y must be greater than or equal to 0");
	ONEC_ASSERT(offset.z >= 0, "Offset z must be greater than or equal to 0");
	ONEC_ASSERT(size.x >= 0, "Size x must be greater than or equal to 0");
	ONEC_ASSERT(size.y >= 0, "Size y must be greater than or equal to 0");
	ONEC_ASSERT(size.z >= 0, "Size z must be greater than or equal to 0");

	cudaMemcpy3DParms copyParameters{};
	copyParameters.extent = cudaExtent{ static_cast<size_t>(size.x), static_cast<size_t>(size.y), static_cast<size_t>(size.z) };

	const size_t stride{ static_cast<size_t>((m_format.x + m_format.y + m_format.z + m_format.w) / 8) };
	const size_t pitch{ copyParameters.extent.width * stride };

	copyParameters.srcPtr = make_cudaPitchedPtr(const_cast<std::byte*>(source.getData()), pitch, copyParameters.extent.width, copyParameters.extent.height);
	copyParameters.dstArray = m_handle;
	copyParameters.dstPos = cudaPos{ static_cast<size_t>(offset.x), static_cast<size_t>(offset.y), static_cast<size_t>(offset.z) };
	copyParameters.kind = cudaMemcpyHostToDevice;

	CU_CHECK_ERROR(cudaMemcpy3D(&copyParameters));
}

void Array::download(const Span<std::byte>&& destination, const glm::ivec3 size) const
{
	download(std::forward<const Span<std::byte>&&>(destination), glm::ivec3{ 0 }, size);
}

void Array::download(const Span<std::byte>&& destination, const glm::ivec3 offset, const glm::ivec3 size) const
{
	ONEC_ASSERT(offset.x >= 0, "Offset x must be greater than or equal to 0");
	ONEC_ASSERT(offset.y >= 0, "Offset y must be greater than or equal to 0");
	ONEC_ASSERT(offset.z >= 0, "Offset z must be greater than or equal to 0");
	ONEC_ASSERT(size.x >= 0, "Size x must be greater than or equal to 0");
	ONEC_ASSERT(size.y >= 0, "Size y must be greater than or equal to 0");
	ONEC_ASSERT(size.z >= 0, "Size z must be greater than or equal to 0");

	cudaMemcpy3DParms copyParameters{};
	copyParameters.srcArray = m_handle;
	copyParameters.srcPos = cudaPos{ static_cast<size_t>(offset.x), static_cast<size_t>(offset.y), static_cast<size_t>(offset.z) };
	copyParameters.extent = cudaExtent{ static_cast<size_t>(size.x), static_cast<size_t>(size.y), static_cast<size_t>(size.z) };

	const size_t stride{ static_cast<size_t>((m_format.x + m_format.y + m_format.z + m_format.w) / 8) };
	const size_t pitch{ copyParameters.extent.width * stride };

	copyParameters.dstPtr = make_cudaPitchedPtr(destination.getData(), pitch, copyParameters.extent.width, copyParameters.extent.height);
	copyParameters.kind = cudaMemcpyDeviceToHost;

	CU_CHECK_ERROR(cudaMemcpy3D(&copyParameters));
}

cudaArray_const_t Array::getHandle() const
{
	return m_handle;
}

cudaArray_t Array::getHandle()
{
	return m_handle;
}

cudaTextureObject_t Array::getTextureObject()
{
	return m_textureObject;
}

cudaSurfaceObject_t Array::getSurfaceObject()
{
	return m_surfaceObject;
}

glm::ivec3 Array::getSize() const
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
	return m_size.x == 0;
}

void Array::create(const glm::ivec3 size, const cudaChannelFormatDesc& format, const unsigned int flags, const cudaTextureDesc* const textureDescriptor, const bool createSurfaceObject)
{
	if (size != glm::ivec3{ 0 })
	{
		cudaResourceDesc resourceDescriptor{};
		resourceDescriptor.resType = cudaResourceTypeArray;

		const cudaExtent extent{ static_cast<size_t>(size.x), static_cast<size_t>(size.y), static_cast<size_t>(size.z) };
		CU_CHECK_ERROR(cudaMalloc3DArray(&resourceDescriptor.res.array.array, &format, extent, flags));

		m_handle = resourceDescriptor.res.array.array;
		m_graphicsResource = cudaGraphicsResource_t{};
		m_size = size;
		m_format = format;
		m_flags = flags;

		if (textureDescriptor != nullptr)
		{
			CU_CHECK_ERROR(cudaCreateTextureObject(&m_textureObject, &resourceDescriptor, textureDescriptor, nullptr));
		}
		else
		{
			m_textureObject = cudaTextureObject_t{};
		}

		if (createSurfaceObject)
		{
			CU_CHECK_ERROR(cudaCreateSurfaceObject(&m_surfaceObject, &resourceDescriptor));
		}
		else
		{
			m_surfaceObject = cudaSurfaceObject_t{};
		}
	}
	else
	{
		m_handle = cudaArray_t{};
		m_textureObject = cudaTextureObject_t{};
		m_surfaceObject = cudaSurfaceObject_t{};
		m_graphicsResource = cudaGraphicsResource_t{};
		m_size = glm::ivec3{ 0 };
		m_format = cudaCreateChannelDesc<cudaChannelFormatKindNone>();
		m_flags = 0;
	}
}

void Array::create(const std::filesystem::path& file, const unsigned int flags, const cudaTextureDesc* const textureDescriptor, const bool createSurfaceObject)
{
	glm::ivec2 size;
	const auto data{ readImage(file, size, m_format) };

	create(glm::ivec3{ size, 0 }, m_format, flags, textureDescriptor, createSurfaceObject);
	upload({ data.get(), size.x * size.y }, glm::ivec3{ size, 0 });
}

void Array::create(onec::Texture& texture, const cudaTextureDesc* const textureDescriptor, const bool createSurfaceObject)
{
	cudaGraphicsResource_t graphicsResource{ texture.getGraphicsResource() };
	CU_CHECK_ERROR(cudaGraphicsMapResources(1, &graphicsResource));

	cudaResourceDesc resourceDescriptor{};
	resourceDescriptor.resType = cudaResourceTypeArray;

	cudaExtent extent;
	cudaChannelFormatDesc format;
	unsigned int flags;

	CU_CHECK_ERROR(cudaGraphicsSubResourceGetMappedArray(&resourceDescriptor.res.array.array, graphicsResource, 0, 0));
	CU_CHECK_ERROR(cudaArrayGetInfo(&format, &extent, &flags, resourceDescriptor.res.array.array));

	m_handle = resourceDescriptor.res.array.array;
	m_graphicsResource = graphicsResource;
	m_size = glm::ivec3{ extent.width, extent.height, extent.depth };
	m_format = format;
	m_flags = flags;

	if (textureDescriptor != nullptr)
	{
		CU_CHECK_ERROR(cudaCreateTextureObject(&m_textureObject, &resourceDescriptor, textureDescriptor, nullptr));
	}
	else
	{
		m_textureObject = cudaTextureObject_t{};
	}

	if (createSurfaceObject)
	{
		CU_CHECK_ERROR(cudaCreateSurfaceObject(&m_surfaceObject, &resourceDescriptor));
	}
	else
	{
		m_surfaceObject = cudaSurfaceObject_t{};
	}
}

void Array::destroy()
{
	if (m_graphicsResource != cudaGraphicsResource_t{})
	{
		const cudaTextureObject_t textureObject{ m_textureObject };
		const cudaSurfaceObject_t surfaceObject{ m_surfaceObject };

		if (textureObject != cudaTextureObject_t{})
		{
			CU_CHECK_ERROR(cudaDestroyTextureObject(textureObject));
		}

		if (surfaceObject != cudaSurfaceObject_t{})
		{
			CU_CHECK_ERROR(cudaDestroySurfaceObject(surfaceObject));
		}

		CU_CHECK_ERROR(cudaGraphicsUnmapResources(1, &m_graphicsResource));
	}
	else if (!isEmpty())
	{
		const cudaTextureObject_t textureObject{ m_textureObject };
		const cudaSurfaceObject_t surfaceObject{ m_surfaceObject };

		if (textureObject != cudaTextureObject_t{})
		{
			CU_CHECK_ERROR(cudaDestroyTextureObject(textureObject));
		}

		if (surfaceObject != cudaSurfaceObject_t{})
		{
			CU_CHECK_ERROR(cudaDestroySurfaceObject(surfaceObject));
		}

		CU_CHECK_ERROR(cudaFreeArray(m_handle));
	}
}

}
}
