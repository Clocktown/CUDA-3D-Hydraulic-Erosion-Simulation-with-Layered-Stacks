#include "surface.hpp"
#include "array.hpp"
#include "../config/cu.hpp"
#include <cuda_runtime.h>
#include <utility>

namespace onec
{
namespace cu
{

Surface::Surface() :
	m_handle{ cudaSurfaceObject_t{} }
{

}

Surface::Surface(Array& array)
{
	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
								     .res{ .array{ .array{ array.getHandle() } } } };

	CU_CHECK_ERROR(cudaCreateSurfaceObject(&m_handle, &resource));
}

Surface::Surface(Surface&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, cudaSurfaceObject_t{}) }
{

}

Surface::~Surface()
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaDestroySurfaceObject(m_handle));
	}
}

Surface& Surface::operator=(Surface&& other) noexcept
{
	if (this != &other)
	{
		if (!isEmpty())
		{
			CU_CHECK_ERROR(cudaDestroySurfaceObject(m_handle));
		}

		m_handle = std::exchange(other.m_handle, cudaSurfaceObject_t{});
	}

	return *this;
}

void Surface::initialize(Array& array)
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaDestroySurfaceObject(m_handle));
	}

	const cudaResourceDesc resource{ .resType{ cudaResourceTypeArray },
							         .res{ .array{ .array{ array.getHandle() } } } };

	CU_CHECK_ERROR(cudaCreateSurfaceObject(&m_handle, &resource));
}

void Surface::release()
{
	if (!isEmpty())
	{
		CU_CHECK_ERROR(cudaDestroySurfaceObject(m_handle));
		m_handle = cudaSurfaceObject_t{};
	}
}

cudaSurfaceObject_t Surface::getHandle()
{
	return m_handle;
}

bool Surface::isEmpty() const
{
	return m_handle == cudaSurfaceObject_t{};
}

}
}
