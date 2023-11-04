#pragma once

#include "array.hpp"
#include <cuda_runtime.h>

namespace onec
{
namespace cu
{

class Surface
{
public:
	explicit Surface();
	explicit Surface(Array& array);
	Surface(const Surface& other) = delete;
	Surface(Surface&& other) noexcept;

	~Surface();

	Surface& operator=(const Surface& other) = delete;
	Surface& operator=(Surface&& other) noexcept;

	void initialize(Array& array);
	void release();

	cudaSurfaceObject_t getHandle();
	bool isEmpty() const;
private:
	cudaSurfaceObject_t m_handle;
};

}
}
