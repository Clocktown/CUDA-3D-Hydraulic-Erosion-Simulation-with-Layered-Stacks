#include "texture.hpp"
#include "array.hpp"
#include "array_view.hpp"
#include "../config/cu.hpp"
#include <cuda_runtime.h>
#include <utility>

namespace onec
{
namespace cu
{

Texture::Texture() :
	m_handle{ cudaTextureObject_t{} }
{

}

Texture::Texture(const Array& array, const cudaTextureDesc& desc)
{
	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
								     .res{ .array{ .array{ const_cast<Array&>(array).getHandle() } } } };

	CU_CHECK_ERROR(cudaCreateTextureObject(&m_handle, &resource, &desc, nullptr));
}

Texture::Texture(const ArrayView& arrayView, const cudaTextureDesc& desc)
{
	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
								     .res{ .array{ .array{ const_cast<ArrayView&>(arrayView).getHandle() } } } };

	CU_CHECK_ERROR(cudaCreateTextureObject(&m_handle, &resource, &desc, nullptr));
}

Texture::Texture(Texture&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, cudaTextureObject_t{}) }
{

}

Texture::~Texture()
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaDestroyTextureObject(m_handle));
	}
}

Texture& Texture::operator=(Texture&& other) noexcept
{
	if (this != &other)
	{
		if (!isEmpty())
		{
			CU_CHECK_ERROR(cudaDestroyTextureObject(m_handle));
		}

		m_handle = std::exchange(other.m_handle, cudaTextureObject_t{});
	}

	return *this;
}

void Texture::initialize(const Array& array, const cudaTextureDesc& desc)
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaDestroyTextureObject(m_handle));
	}

	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
									 .res{ .array{ .array{ const_cast<Array&>(array).getHandle() } } } };

	CU_CHECK_ERROR(cudaCreateTextureObject(&m_handle, &resource, &desc, nullptr));
}

void Texture::initialize(const ArrayView& arrayView, const cudaTextureDesc& desc)
{
		if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaDestroyTextureObject(m_handle));
	}

	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
									 .res{ .array{ .array{ const_cast<ArrayView&>(arrayView).getHandle() } } } };

	CU_CHECK_ERROR(cudaCreateTextureObject(&m_handle, &resource, &desc, nullptr));
}

void Texture::release()
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaDestroyTextureObject(m_handle));
		m_handle = cudaTextureObject_t{};
	}
}

cudaTextureObject_t Texture::getHandle()
{
	return m_handle;
}

bool Texture::isEmpty() const
{
	return m_handle == cudaTextureObject_t{};
}

}
}
